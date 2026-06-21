---
name: rag-improve
description: >
  recsys-coach RAG 파이프라인을 개선한다. 검색 품질, 프롬프트, 청킹, 재순위(reranking),
  하이브리드 검색, 답변 품질 실험을 설계-구현-검증 사이클로 처리한다.
  "RAG 개선", "답변 품질", "검색 품질", "청크", "프롬프트 수정", "재순위", "파이프라인 실험",
  "qa_chain 수정", "다시 실행", "재실행", "업데이트", "이전 결과 개선", "결과 보완" 키워드에 트리거.
---

## 역할

rag-engineer(구현)와 qa-validator(검증)를 조율하는 오케스트레이터.
설계-구현-검증 사이클을 반복하며 RAG 파이프라인 품질을 높인다.

**실행 모드:** 에이전트 팀 (rag-engineer + qa-validator)

## Phase 0: 컨텍스트 확인

시작 시 기존 작업 상태를 먼저 확인한다:

```
_workspace/ 존재 + 부분 수정 요청 → 부분 재실행 (해당 에이전트만 재호출)
_workspace/ 존재 + 새 전략 제공    → 새 실행 (_workspace를 _workspace_prev/로 이동)
_workspace/ 미존재                  → 초기 실행
```

에이전트가 이전 산출물(`_workspace/`)을 읽고 개선점을 반영하도록 컨텍스트를 전달한다.

## Phase 1: 전략 선택

사용자 요청에서 개선 목표를 파악하고, 아래 전략 중 적합한 것을 선택한다.
전략이 불명확하면 사용자에게 1가지만 확인하고 진행한다.

전략 목록과 세부 기준은 `references/strategies.md` 참조.

**주요 전략:**
- A. 청킹 파라미터 조정 (chunk_size, overlap)
- B. 프롬프트 강화 (시스템 프롬프트, 규칙 추가)
- C. 검색 가중치 조정 (BM25:FAISS 비율)
- D. Reranker top_k 조정
- E. 컨텍스트 enrichment 개선 (헤더, 출처 포맷)

## Phase 2: 구현 (rag-engineer)

rag-engineer 에이전트를 호출한다. 전달 사항:
- 선택된 전략 + 구체적 변경 지시
- 현재 `src/qa_chain.py` 상태
- 이전 실험 결과가 있으면 `_workspace/` 경로

구현 완료 후 rag-engineer는 qa-validator에게 직접 검증 요청을 보낸다.

## Phase 3: 검증 (qa-validator)

qa-validator가 테스트를 실행하고 결과를 보고한다.

- **통과**: rag-coach에게 "통과" 보고 → Phase 4로
- **실패**: rag-engineer에게 원인 피드백 → Phase 2 재진입 (최대 2회 재시도)
- **2회 재시도 후 실패**: 결과 없이 Phase 4 진행, 블로커 명시

## Phase 4: 결과 종합 및 보고

사용자에게 보고:
1. 적용된 전략 요약
2. 변경된 코드 (diff 수준)
3. 테스트 결과
4. 다음 시도 권장 전략 (있으면)

중간 산출물은 `_workspace/` 보존. 최종 변경은 `src/qa_chain.py`에 반영.

## 데이터 흐름

```
[rag-coach] → 전략 + 지시 → [rag-engineer]
[rag-engineer] → 구현 완료 알림 → [qa-validator]
[qa-validator] → 검증 결과 → [rag-engineer] (실패 시 피드백)
[qa-validator] → 최종 통과 → [rag-coach]
[rag-coach] → 결과 종합 → [사용자]
```

파일 기반 전달: `_workspace/{phase}_{agent}_{artifact}.md`

## 에러 핸들링

- API 호출 실패: `OFFLINE_MODE=true`로 재시도
- Import 오류: requirements.txt 누락 패키지 명시 후 사용자에게 보고
- 2회 재시도 후 검증 실패: 블로커로 보고, 이전 상태 유지

## 테스트 시나리오

**정상 흐름:**
1. 사용자: "프롬프트 강화해줘"
2. 전략 B 선택 → rag-engineer 호출
3. 구현 완료 → qa-validator 검증
4. 전체 통과 → 결과 보고

**에러 흐름:**
1. qa-validator가 ImportError 발견
2. rag-engineer에게 원인 전달
3. requirements.txt 확인 후 재구현
4. 2차 검증 통과
