# 고려대 실전SW 1차 과제 - LangChain RAG 챗봇

Flask 기반의 경량 웹 UI 위에서 LangChain을 활용한 RAG(Retrieval-Augmented Generation) 챗봇을 구축했습니다. 두 개 이상의 로컬 지식원을 벡터화하여 FAISS 벡터스토어에 저장하고, HuggingFace 임베딩 + OpenAI(ChatGPT) LLM 조합으로 답변을 생성합니다. 답변에는 항상 참조한 문서의 출처가 함께 표기됩니다.

## 프로젝트 구조

```
├── app
│   ├── __init__.py
│   ├── rag_pipeline.py   # LangChain RAG 체인 및 citation 로직
│   └── server.py         # Flask 앱, REST 엔드포인트 및 UI 렌더링
├── data                  # 최소 2개의 도메인 문서 (예: 연구/학생지원)
│   ├── ku_ai_track.md
│   └── ku_student_success.md
├── static
│   └── style.css
├── templates
│   └── index.html
├── requirements.txt
└── README.md
```

## 실행 방법

1. **의존성 설치**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **환경 변수 설정**
   - `.env` 파일 혹은 셸 환경변수로 아래 값을 지정합니다.
     ```
     OPENAI_API_KEY=sk-...
     OPENAI_MODEL=gpt-4o-mini   # (선택) 기본값은 gpt-4o-mini
     EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
     ```
   - OpenAI 대신 다른 LLM(HuggingFaceHub, Ollama 등)을 사용하고 싶다면 `app/rag_pipeline.py`의 `build_llm()` 부분을 수정하면 됩니다.

3. **서버 실행**
   ```bash
   flask --app app/server.py --debug run
   ```
   또는
   ```bash
   python -m app.server
   ```

4. **웹 UI**
   - 브라우저에서 `http://127.0.0.1:5000` 접속
   - 질문 입력 후 전송하면 답변과 함께 참조 문서 리스트가 표시됩니다.

## RAG 파이프라인 개요

- `DirectoryLoader`가 `data/` 폴더 내 모든 `.md` 문서를 읽고 `RecursiveCharacterTextSplitter`가 chunking 합니다.
- `HuggingFaceEmbeddings` (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) 모델로 각 chunk를 임베딩하고, `FAISS`에 저장합니다.
- 사용자가 질문하면 FAISS에서 top-k 문서를 찾아 `ChatPromptTemplate`에 삽입하고, `ChatOpenAI`가 답변을 생성합니다.
- 응답 단계에서 실제로 사용된 문서의 경로/이름을 후처리하여 `출처` 섹션으로 노출합니다.

## 과제 요구사항 만족 여부

- [x] LangChain 기반 RAG 파이프라인
- [x] 임베딩 & 벡터스토어 직접 구성 (HuggingFace Embedding + FAISS)
- [x] 데이터 소스 2개 이상 (`data/` 내 2개 문서)
- [x] 베드락/스트림릿 미사용, Flask UI 제공
- [x] 답변에 출처 표기

## 추가 개선 아이디어

- FastAPI + WebSocket으로 스트리밍 응답 지원
- 지식원을 S3, Google Drive 등 외부 스토리지로 확장
- LangSmith/Weights & Biases를 사용한 프롬프트 및 체인 모니터링
- 사용자 피드백(thumbs up/down)을 저장하여 재학습/재정렬

행운을 빕니다! 🎓💻

