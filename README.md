# 🏆 RecSys Coach

추천 시스템 대회 참가자를 위한 AI 코칭 도우미.  
대회 문서 Q&A와 자동 EDA 리포트 기능을 하나의 Streamlit 앱에서 제공합니다.

---

## 주요 기능

### 대회 문서 Q&A
- PDF·TXT 파일을 업로드하면 즉시 벡터 인덱스를 구성합니다.
- 질문을 입력하면 관련 문서 청크를 검색해 LLM이 한국어로 답변합니다.
- 답변 하단에 출처 파일명과 페이지를 표시합니다.

### 자동 EDA 리포트
- CSV 파일(업로드 또는 서버 경로)을 입력받아 자동으로 EDA 리포트를 생성합니다.
- 기초 통계, 결측값, 상관관계 히트맵, 시계열 차트, 값 분포 표를 포함합니다.
- GPU(NVIDIA CUDA)가 감지되면 cuDF 가속 경로를, 없으면 pandas 경로를 사용합니다.

---

## 기술 스택

| 구분 | 내용 |
|------|------|
| 언어 | Python 3.10+ |
| UI | Streamlit |
| RAG | LangChain (LCEL) + FAISS |
| 임베딩 | Upstage `solar-embedding-1-large` |
| LLM (기본) | Upstage Solar Pro |
| LLM (로컬) | Qwen2.5-7B-Instruct / Llama-3 계열 (HuggingFace) |
| 검색 | BM25 + FAISS 앙상블 + Cross-Encoder Reranker |
| EDA (CPU) | pandas + Plotly |
| EDA (GPU) | cuDF (RAPIDS) + Plotly |

---

## 설치

```bash
# 1. 저장소 클론 후 가상환경 생성
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 UPSTAGE_API_KEY 등을 입력하세요
```

> **GPU(cuDF) 가속 사용 시** `requirements.txt`의 주석을 참고해 cudf-cu12를 별도 설치하세요.
> 일반 CPU 환경에서는 별도 설치 없이 동작합니다.

---

## 환경 변수

`.env.example`을 복사해 `.env`를 만들고 아래 값을 채워 넣습니다.

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `UPSTAGE_API_KEY` | Upstage API 키 (필수) | — |
| `OFFLINE_MODE` | `true`로 설정 시 API 호출 없이 로컬 검색 사용 | `false` |
| `LLM_BACKEND` | `upstage` / `qwen` / `llama3` | `upstage` |
| `HF_TOKEN` | HuggingFace 토큰 (로컬 LLM 사용 시) | — |
| `LOCAL_MODEL_ID` | 로컬 모델 ID | `Qwen/Qwen2.5-7B-Instruct` |
| `LLAMA_LOAD_IN_8BIT` | 8-bit 양자화 활성화 여부 | `true` |
| `LLAMA_TEMPERATURE` | 생성 온도 | `0.3` |
| `LLAMA_MAX_NEW_TOKENS` | 최대 생성 토큰 수 | `512` |

---

## 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`이 자동으로 열립니다.

---

## LLM 백엔드 선택

| 모드 | `.env` 설정 | 설명 |
|------|-------------|------|
| Upstage Solar (기본) | `LLM_BACKEND=upstage` | API 키 필요. 가장 안정적인 한국어 품질 |
| Qwen2.5 (로컬) | `LLM_BACKEND=qwen` | GPU 권장. HF 모델 다운로드 필요 |
| Llama-3 (로컬) | `LLM_BACKEND=llama3` | GPU 권장. HF 모델 다운로드 필요 |
| 오프라인 | `OFFLINE_MODE=true` | LLM 없이 검색 결과만 반환. 개발·테스트 용도 |

---

## 아키텍처

```
app.py                  ← Streamlit 진입점 (사이드바 탭 라우팅)
├── src/qa_chain.py     ← RAG 파이프라인
│   ├── build_vectorstore()   : 문서 청킹 → 임베딩 → FAISS 인덱스
│   └── answer_question()     : BM25+FAISS 검색 → Cross-Encoder Rerank → LLM
└── src/eda.py          ← EDA 리포트
    ├── _eda_gpu()      : cuDF 경로
    └── _eda_cpu()      : pandas 경로
```

**RAG 파이프라인 흐름**

```
문서 업로드
  → 청킹 (RecursiveCharacterTextSplitter, 600자 / 80자 오버랩)
  → 출처 헤더 prefix 추가
  → Upstage Embeddings (solar-embedding-1-large)
  → FAISS 벡터스토어

질문 입력
  → BM25 + FAISS 앙상블 검색 (k=6)
  → Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) → 상위 3청크
  → Solar Pro / 로컬 LLM 생성
  → 답변 + 출처 반환
```

---

## 테스트

```bash
# 전체 테스트
python -m pytest tests/ -v

# 단일 테스트
python -m pytest tests/test_qa_chain.py::test_name -v
```

---

## 주요 트러블슈팅

### Q&A 답변이 `[object Object]`로 깨지던 버그

코드 블록이 포함된 답변이 `[object Object]` 박스로 렌더되던 문제를 **근본 원인 수준에서** 해결했습니다.

- **원인:** 마크다운을 HTML로 재구현한 `_md_to_html`(165줄)의 출력을 `st.markdown(..., unsafe_allow_html=True)`로 주입했는데, Streamlit이 그 HTML을 자체 렌더 파이프라인으로 재파싱하다 코드 블록을 깨뜨려 JS가 `[object Object]`로 강제변환.
- **해결:** Streamlit이 기본 제공하는 `st.markdown(text)` 마크다운 렌더링으로 교체하고 `_md_to_html` 및 의존성(markdown-it-py, pygments)을 제거. **net −113줄**, 의존성 2개 감소.
- **재발 방지:** LLM 출력 노이즈 패턴에 대한 회귀 테스트 5건 추가. (상세: [PORTFOLIO.md](PORTFOLIO.md) §4, [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md))

> 증상이 난 경로만 패치하지 않고 렌더러(근본 원인)를 고쳐, 버그 수정과 코드 감축을 동시에 달성한 사례입니다.

---

## 주의사항

- `.env`, `*.csv`, `.streamlit/secrets.toml` 파일은 절대 커밋하지 마세요.
- 대용량 CSV(200MB 초과)는 UI의 **서버 경로 직접 입력** 모드를 사용하세요.
- Upstage API 크레딧을 아끼려면 `OFFLINE_MODE=true`로 개발하세요.
