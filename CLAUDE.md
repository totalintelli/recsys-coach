# CLAUDE.md

이 파일은 저장소에서 작업할 때 Claude Code(claude.ai/code)에 제공되는 안내 문서입니다.

## 프로젝트

**RecSys Coach** — 추천 시스템 대회 참가자를 위한 AI 코칭 도우미. LangChain + Streamlit + Upstage Solar LLM API로 구축됩니다.

## 환경 설정

```bash
cp .env.example .env          # UPSTAGE_API_KEY 입력
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 명령어

```bash
streamlit run app.py                                    # 앱 실행
python -m pytest tests/ -v                              # 전체 테스트 실행
python -m pytest tests/path/to/test_file.py::test_name -v  # 단일 테스트 실행
```

`.env`에서 `OFFLINE_MODE=true`로 설정하면 Upstage API 호출을 건너뛰고 로컬 검색 폴백을 사용합니다 (API 크레딧 소모 없이 개발할 때 유용).

## 아키텍처

- **`app.py`** — Streamlit 진입점. 탭 전환은 `st.sidebar.radio()`로 구현 (`st.tabs()` 대신 — `st.chat_input()`이 탭 내부에서 최상단에 렌더링되는 Streamlit 버그 우회). `st.chat_input()`은 최상위 레벨에 배치해 뷰포트 하단 고정. EDA 탭은 파일 업로드(200MB 제한)와 서버 경로 직접 입력 두 가지 모드를 지원.
- **`src/qa_chain.py`** — RAG 파이프라인. `build_vectorstore()`는 UpstageEmbeddings + FAISS + 청크 출처 헤더 prefix. `answer_question()`은 BM25+FAISS 앙상블 리트리버 → Cross-Encoder reranking(상위 3개) → LCEL 체인 구조. LLM 백엔드는 `LLM_BACKEND` 환경변수로 분기(`upstage` → Solar-Pro | `llama3`/`qwen` → HuggingFacePipeline 8bit). `OFFLINE_MODE=true` 시 LLM 호출 없이 검색 컨텍스트를 직접 반환. 응답 캐시(`_answer_cache`) 내장. `langchain.chains.RetrievalQA` 미사용.
- **`src/eda.py`** — GPU/CPU 분기 EDA. `Checklist` 클래스로 □/✓ 실시간 진행 표시. `detect_compute_backend()`가 cuDF + torch.cuda로 GPU 감지. GPU면 cuDF + Plotly, CPU면 pandas + plotly 직접 분석(ydata-profiling 미사용). 수치 컬럼을 연속형/범주형ID/식별자ID 세 그룹으로 분류해 각각 히스토그램·막대·고유값개수 요약으로 표시. 히스토그램은 서버에서 빈 사전 집계(MessageSizeError 방지).
- **데이터 파일** (`*.csv`)은 gitignore 처리됨 — 로컬에서 저장소 루트 또는 `data/` 디렉토리에 배치.

## Import 주의사항 (LangChain 버전 호환)

최신 LangChain에서 이동된 패키지들 — 아래 경로를 사용해야 함:

- `langchain.schema.Document` → `langchain_core.documents.Document`
- `langchain.text_splitter` → `langchain_text_splitters`
- `langchain.chains.RetrievalQA` → 제거됨, LCEL로 대체
- `langchain_text_splitters` 별도 설치 필요: `requirements.txt`에 `langchain-text-splitters>=0.2` 포함

## 민감한 파일 — 절대 커밋 금지

- `.env` (API 키)
- `*.csv` (대회 데이터)
- `.streamlit/secrets.toml`
