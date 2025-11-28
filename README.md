
# 고려대 실전SW 1차 과제 - LangChain RAG 챗봇

Flask 기반의 경량 웹 UI 위에서 LangChain을 활용한 RAG(Retrieval-Augmented Generation) 챗봇을 구축했습니다. 두 개 이상의 로컬 지식원을 벡터화하여 FAISS 벡터스토어에 저장하고, HuggingFace 임베딩 + OpenAI(ChatGPT) LLM 조합으로 답변을 생성합니다. 답변에는 항상 참조한 문서의 출처가 함께 표기됩니다.

## 프로젝트 구조

```
├── app
│   ├── __init__.py
│   ├── rag_pipeline.py   # LangChain RAG 체인 및 citation 로직
│   └── server.py         # Flask 앱, REST 엔드포인트 및 UI 렌더링
├── data
│   └── feeds/            # RSS에서 동적으로 적재되는 문서
├── feeds.yaml            # 수집할 RSS 피드 정의
├── static
│   └── style.css
├── templates
│   └── index.html
├── scripts
│   └── fetch_feeds.py
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

## RSS 피드 연동으로 데이터 확장하기

`feeds.yaml`에 원하는 RSS 주소를 등록하고 `scripts/fetch_feeds.py`를 실행하면 최신 기사들이 `data/feeds/` 폴더에 마크다운으로 누적됩니다. `DirectoryLoader`는 `data/` 아래 모든 `.md` 파일을 자동으로 읽으므로, 외부 뉴스/보도자료도 같은 RAG 파이프라인으로 검색됩니다.

1. 피드 설정 편집
   ```yaml
   feeds:
     - name: "Korea University News"
       url: "https://www.korea.ac.kr/rss/RSS.jsp?part=univ_news"
       limit: 5
     - name: "Ministry of Education Announcements"
       url: "https://www.moe.go.kr/boardCnts/Rss.do?boardID=294"
       limit: 5
   ```

2. 스크립트 실행
   ```bash
   .\.venv\Scripts\activate
   python scripts/fetch_feeds.py --config feeds.yaml --output data/feeds
   ```

3. `data/feeds/*.md`가 생성되면 Flask 서버를 재시작하여 새 지식원이 반영된 챗봇을 바로 사용할 수 있습니다.

