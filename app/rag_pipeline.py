from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _default_data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "data")).resolve()


@dataclass
class RagSettings:
    """Configurable settings for the RAG pipeline."""

    data_dir: Path = field(default_factory=_default_data_dir)
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))


class RagChatbot:
    """Encapsulates LangChain-based RAG workflow with citation support."""

    def __init__(self, settings: RagSettings | None = None) -> None:
        self.settings = settings or RagSettings()
        self._validate_data_dir()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.settings.embedding_model)
        self.vectorstore = self._build_vectorstore()
        self.prompt = self._build_prompt()
        self.llm = self._build_llm()

    def _validate_data_dir(self) -> None:
        if not self.settings.data_dir.exists():
            raise FileNotFoundError(
                f"데이터 디렉터리({self.settings.data_dir})를 찾을 수 없습니다. "
                "data/ 폴더를 생성하고 최소 2개 이상의 문서를 넣어주세요."
            )

    def _load_documents(self):
        loader = DirectoryLoader(
            str(self.settings.data_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        documents = loader.load()
        if not documents:
            raise RuntimeError("데이터 폴더에 로드 가능한 문서가 없습니다.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n## ", "\n", " ", ""],
        )
        return splitter.split_documents(documents)

    def _build_vectorstore(self) -> FAISS:
        chunks = self._load_documents()
        return FAISS.from_documents(chunks, self.embeddings)

    def _build_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.openai_model,
            temperature=self.settings.temperature,
        )

    def _build_prompt(self) -> ChatPromptTemplate:
        system_message = (
            "너는 고려대학교 실전SW 과제를 돕는 RAG 챗봇이다. "
            "제공된 콘텍스트 안에서만 답변하고, 모르는 내용은 솔직하게 모른다고 말해라. "
            "답변 마지막에는 [source_id] 형태로 최소 1개 이상의 출처를 표기하라."
        )
        template = (
            "질문: {question}\n"
            "-----------------\n"
            "콘텍스트:\n{context}\n"
            "-----------------\n"
            "위 자료만 활용해 concise 하게 대답해라."
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", template),
            ]
        )

    def _format_context(self, docs) -> str:
        formatted = []
        for idx, doc in enumerate(docs, start=1):
            source = self._format_source_name(doc.metadata.get("source", f"chunk-{idx}"))
            formatted.append(f"[{idx}:{source}] {doc.page_content}")
        return "\n\n".join(formatted)

    @staticmethod
    def _format_source_name(source_path: str) -> str:
        return Path(source_path).stem.replace("_", " ")

    def _extract_sources(self, docs) -> List[Dict[str, str]]:
        seen = {}
        for doc in docs:
            raw = doc.metadata.get("source", "unknown")
            name = self._format_source_name(raw)
            seen[name] = str(Path(raw).name)
        return [{"title": title, "path": path} for title, path in seen.items()]

    def answer(self, question: str) -> Dict[str, object]:
        if not question.strip():
            raise ValueError("질문이 비어 있습니다.")

        docs = self.vectorstore.similarity_search(question, k=self.settings.top_k)
        context = self._format_context(docs)
        chain = self.prompt | self.llm
        llm_response = chain.invoke({"question": question, "context": context})

        sources = self._extract_sources(docs)
        citation_text = ", ".join(f"[{item['title']}]" for item in sources) or "[no-source]"
        full_answer = f"{llm_response.content}\n\n출처: {citation_text}"

        return {
            "answer": full_answer,
            "sources": sources,
        }

