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

- **`app.py`** — Streamlit 진입점. 탭 전환은 `st.sidebar.radio()`로 구현 (`st.tabs()` 대신 — `st.chat_input()`이 탭 내부에서 최상단에 렌더링되는 Streamlit 버그 우회). `st.chat_input()`은 최상위 레벨에 배치해 뷰포트 하단 고정.
- **`src/qa_chain.py`** — RAG 파이프라인. `build_vectorstore()`는 UpstageEmbeddings + FAISS, `answer_question()`은 LCEL 체인(`retriever | prompt | ChatUpstage | StrOutputParser`) 사용. `langchain.chains.RetrievalQA` 미사용.
- **`src/eda.py`** — GPU/CPU 분기 EDA. `torch.cuda.is_available()`로 백엔드 감지 후 GPU면 cuDF + Plotly, CPU면 ydata-profiling full 모드 (`minimal=False`).
- **이중 모드 RAG 파이프라인**: 온라인 모드는 LangChain을 통해 Upstage Solar LLM 사용; 오프라인 모드는 로컬 키워드 검색으로 폴백.
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

## 응답 규칙

- 수정 요청을 완료한 후 **반드시** 해당 수정에서 변경된 파일 목록만 채팅창에 표시한다. 전체 세션 이력이 아닌 방금 완료한 요청 기준으로만 표시한다.

---

## 하네스: RecSys Coach RAG

**목표:** RAG 파이프라인 개선 실험을 설계-구현-검증 사이클로 자동화한다.

**트리거:** RAG 개선, 검색 품질, 답변 품질, 프롬프트 수정, 청킹, 재순위, qa_chain 수정, 테스트 실행 요청 시 `rag-improve` 스킬을 사용하라. 단순 질문은 직접 응답 가능.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-06-20 | 초기 구성 | 전체 | rag-engineer + qa-validator 팀 + rag-improve/test-validate 스킬 |
