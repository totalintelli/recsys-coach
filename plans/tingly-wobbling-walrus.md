# Plan: 변경 사항 커밋

## Context

브랜치 `rag-architecture-advanced`에서 Plan A + Plan B 구현이 완료된 상태. 아래 6개 파일을 스테이징 후 커밋한다.

---

## 커밋 대상 파일

| 파일 | 상태 | 내용 |
|------|------|------|
| `src/qa_chain.py` | modified | Plan A + Plan B 전체 RAG 개선 |
| `requirements.txt` | modified | rank-bm25, sentence-transformers 등 패키지 추가 |
| `tests/test_qa_chain.py` | modified | Plan A/B 테스트 클래스 추가 |
| `.env.example` | modified | LLM_BACKEND, HF_TOKEN 등 환경변수 추가 |
| `tests/__init__.py` | untracked | 신규 (빈 파일) |
| `plans/tingly-wobbling-walrus.md` | untracked | 계획 파일 |

---

## 커밋 명령

```bash
git add src/qa_chain.py requirements.txt tests/test_qa_chain.py .env.example tests/__init__.py plans/tingly-wobbling-walrus.md
git commit -m "$(cat <<'EOF'
exp : RAG 답변 품질 개선 — Plan A/B 적용

Plan A: 강화된 프롬프트(역할 정의 + 4가지 규칙), BM25+FAISS 하이브리드 검색,
        문서 노이즈 전처리(_clean_text), 출처 헤더 주입(_enrich_chunk)

Plan B: Cross-Encoder Reranker(k=6→top-3), 답변 캐시(_answer_cache),
        문단 경계 기반 Semantic Chunking(chunk_size 600), LLM 출력 후처리(_clean_answer)

- app.py 수정 없음 (build_vectorstore/answer_question 시그니처 유지)
- sentence-transformers, rank-bm25 미설치 시 graceful fallback
- 단위 테스트 6개 클래스, ~24개 테스트 메서드 추가
EOF
)"
```

---

## 검증

커밋 후 `git log --oneline -3`으로 커밋 생성 확인.
