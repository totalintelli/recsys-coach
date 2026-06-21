---
name: test-validate
description: >
  recsys-coach 프로젝트의 테스트를 실행하고 결과를 해석한다.
  RAG 파이프라인(qa_chain.py) 변경 후 회귀를 확인하거나, 테스트 실패 원인을 분석하거나,
  특정 테스트만 골라 돌리거나, 테스트를 새로 추가해야 할 때 반드시 이 스킬을 사용하라.
  "테스트", "pytest", "테스트 돌려줘", "회귀 확인", "테스트 실패", "test_qa_chain" 키워드에 트리거.
---

## 실행 기준

프로젝트 루트에서 실행한다. 기본은 OFFLINE_MODE=true로 API 소모 없이 검증한다.

```bash
# 전체 테스트
OFFLINE_MODE=true python -m pytest tests/ -v

# 단일 테스트
OFFLINE_MODE=true python -m pytest tests/test_qa_chain.py::test_name -v

# API 포함 (크레딧 소모)
python -m pytest tests/ -v
```

## 결과 해석 원칙

1. **실패 시**: traceback 전체를 읽고 오류 유형을 분류한다
   - `ImportError` → requirements.txt 누락 패키지 명시
   - `AssertionError` → 실제값 vs 기댓값 비교, 로직 버그 추적
   - `API 오류` → 환경 문제(키 없음)와 로직 문제 구분
2. **통과 시**: 몇 개 통과했는지 + 경고(warning) 있으면 함께 보고한다
3. 테스트 자체가 잘못된 assertion이면 "테스트 코드 수정 필요"로 분리 보고한다

## 테스트 추가 가이드

`tests/test_qa_chain.py`에 새 테스트를 추가할 때:
- OFFLINE_MODE 분기를 고려한다 (API 호출 테스트는 `@pytest.mark.skipif(OFFLINE_MODE, ...)`)
- `build_vectorstore()` 테스트는 최소 Document 1개로 smoke test 수준이면 충분하다
- `answer_question()` 테스트는 반환 dict 구조(`answer`, `sources`) 검증을 포함한다

## 테스트 시나리오

- 정상 흐름: `OFFLINE_MODE=true python -m pytest tests/ -v` → 전체 통과
- 회귀 탐지: qa_chain.py 수정 후 재실행 → 변경 전과 결과 비교
