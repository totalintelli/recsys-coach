# 실험 로그 — Commerce Purchase Behavior Prediction

대회: Upstage AI Stages, 비공개(AI부트캠프 24기). 기간 2026-06-24 ~ 2026-07-14.
태스크: train 전체 유저(638,257명)에게 각 10개 아이템 추천. 지표 **NDCG@10 (binary relevance, IDCG=full_k)**.
작업 폴더: `ocr-rs-ad-rs_syd/` (스크립트: local_cv / make_submission / log_lb).

---

## 평가지표 — 반드시 full_k

대회 평가방법 PDF(`raw/평가방법.pdf`) 예시로 확인: 정답이 1개여도 IDCG는 k개 위치를 모두 1로 채움.

```
IDCG@3 = 1/log(1+1) + 1/log(2+1) + 1/log(3+1)   # 정답 1개여도 3개 위치 다 채움
```

→ `local_cv.py`의 `idcg_mode` 기본값을 `standard` → **`full_k`로 변경**(evaluate 시그니처 + CLI).
표준 NDCG(standard, 정답 수만큼만 채움)는 점수를 크게 부풀리므로 비교에 쓰지 말 것.

## CV 설계 (시간 홀드아웃)

test가 "미래 1주 구매(3/1~3/7)"라 랜덤 K-fold는 미래 누수로 LB와 어긋남.
→ 컷오프 T 이전으로만 추천 → T 이후 1주 실제 purchase와 비교. 정답 유저 = 검증 주 구매자 中 T 이전 이력 보유자.
구매가 후반 집중이라 **마지막 주(2/23~2/29, 정답 928명) 단일 폴드만 유효**. 과거 창은 정답 11~55명으로 비어 못 씀.

---

## 모델 비교 (마지막 주 홀드아웃 928유저, full_k NDCG@10)

| 모델 | NDCG@10 | Recall@10 | HitRate@10 | 비고 |
|---|---|---|---|---|
| **covisit** | **0.0731** | 0.4207 | 0.4429 | personal + co-visitation 블렌딩. 현재 최선 |
| personal | 0.0694 | 0.3982 | 0.4203 | 본인 이력 가중·시간감쇠 재랭킹 |
| ALS (w_decay, f32) | 0.0462 | 0.2951 | 0.3125 | ALS 중 최고 설정. covisit의 63% |
| ALS (baseline ones) | 0.0426 | 0.2608 | 0.2780 | train_als.py 기본(label=1) |
| popularity | 0.0331 | 0.1841 | 0.1961 | 전역 인기 |

핵심: 이 대회 정답 대부분이 **본 아이템 재구매**라, 이력을 직접 점수화하는 covisit/personal이 강함.
이벤트 가중 = {view:1, cart:3, purchase:10}, 시간감쇠 14일.

## ALS 검증 (접음)

implicit ALS를 세 갈래로 시도, 전부 covisit 미달:
- 이벤트 가중 단독: 0.0426 → 0.0427 (효과 0. implicit alpha=10이 이미 값 증폭).
- alpha 40으로 ↑: 0.0388 (하락, 노이즈 과신뢰).
- 시간감쇠: 0.0462 (ALS 최고, 그래도 covisit의 63%).
- factors 64: 0.0456.
- **covisit+ALS RRF 블렌딩**: 반반 0.0585 → covisit 가중↑일수록 단조 상승(wc=5 0.0675), 전 구간 covisit 단독 미만. ALS 섞으면 무조건 손해.

→ ALS 라인 종료. `output/output-001.csv`(전체학습 ALS 제출본) 활용 금지.

---

## 제출 기록 (LB)

| 제출일 | 파일 | 모델 | full_k CV | LB Public | 비고 |
|---|---|---|---|---|---|
| 2026-06-24 | exp-003-covisit.csv | covisit | 0.0731 | **0.1544** | 첫 제출 |

**CV-LB 괴리(미해결):** full_k CV 0.073 vs LB 0.154 (LB가 2배 높음). 검증창 바꿔도 수렴 안 함(과거 창 비어서).
가장 합리적 가설: 검증 주(2/23~2/29)와 test 주(3/1~3/7)의 구매자 분포 차이. train만으로 검증 불가(test 주 데이터 없음).
→ **full_k CV는 절대값은 LB와 어긋나도 모델 간 상대 순위는 유효**. "어느 모델이 나은가" 판단에 그대로 사용.

---

## 성능 최적화 (제출 생성 속도)

GPU는 답 아님 — 병목이 행렬곱이 아니라 pandas groupby + 파이썬 dict 루프(CPU·단일코어 바운드).

- **personal**: 유저 for 루프 → numpy-slice. 출력 바이트 동일. 제출 ~28초.
- **covisit**: 유저별 dict 누적 → factorize+numpy. + 이웃 top_nbr=100 컷(인기 아이템 이웃 최대 2.4만 개가 병목). **약 3시간 → 264초(4분 24초)**. top_nbr=100은 NDCG 동등/약간 상승(노이즈 이웃 제거).
- 주의: covisit은 이웃 크기가 데이터 양에 비선형이라 **유저 표본으로 시간 추정 금지** — 전체 실행/홀드아웃으로 직접 측정.

---

## 코칭 앱 — `[object Object]` 렌더 버그 (해결)

증상: Q&A 답변의 코드 블록이 `[object Object]` 회색 박스로 깨짐. 앞선 2커밋의 증상 패치(중첩 `<pre>` 제거, 빈 펜스 제거)로도 재발.

근본 원인: `_md_to_html`(markdown-it-py+pygments로 마크다운→HTML 재구현)의 출력을 `st.markdown(html, unsafe_allow_html=True)`로 주입.
Streamlit이 주입 HTML을 자체 React 파이프라인으로 재파싱하며 인라인 스타일 코드 블록을 노드로 변환 실패 → JS가 객체를 `[object Object]` 문자열로 강제변환.
함수 출력 HTML 자체는 정상이었음(렌더러가 병목). `[object Object]`는 JS 객체→문자열 시그니처.

해결: `_md_to_html`(169줄) 삭제, 호출부 2곳을 `st.markdown(text)`로 교체(Streamlit 기본 마크다운이 펜스 코드·표·목록 렌더링 제공).
의존성 markdown-it-py, pygments 제거. `_clean_answer`(LLM 노이즈 제거)는 답변 경로에 독립적이라 유지.

| 항목 | 값 |
|---|---|
| 코드 변화 | **net −113줄** (추가 170 / 삭제 283, 4파일) |
| 제거 의존성 | markdown-it-py, pygments |
| 회귀 테스트 추가 | 5건 (빈 펜스 / 미닫힘 펜스 / 리스트 내부 펜스 / 노이즈 박스) |
| 테스트 결과 | 38 passed |

회귀 방지 테스트: `test_empty_codeblock_removed`, `test_unclosed_empty_codeblock_does_not_eat_following_text`,
`test_real_codeblock_preserved_with_closing_fence`, `test_object_noise_in_list_item_no_box`, `test_real_codeblock_in_list_preserved`.

---

## 다음 할 일

- [ ] covisit 파라미터 튜닝: n_seed, alpha, top_nbr, decay_days (full_k CV 기준).
- [ ] personal 단독 제출해 LB-CV 순위 보존 확인(CV 신뢰도 검증).
- [ ] covisit 후보 확장(2-hop co-visit 등) — 단 본 아이템 재구매 신호 우선 유지.
- ALS·블렌딩은 천장 확인됨, 재시도 불필요.
