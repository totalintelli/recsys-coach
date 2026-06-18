# 플랜: 채팅 입력창을 항상 최하단에 고정 (탭 내부 버그 우회)

## Context

스크린샷에서 확인된 문제: `st.chat_input()`이 채팅 메시지 **위에** 렌더링된다.

**근본 원인**: Streamlit의 알려진 버그 — `st.tabs()` 블록 **안**에서 `st.chat_input()`을 사용하면
입력창이 탭 콘텐츠 영역 최상단에 렌더링된다 (뷰포트 하단 고정 대신).

**해결 방향**: `st.chat_input()`을 `with tab_qa:` 블록 **바깥**으로 이동시키고,
`st.session_state`의 현재 활성 탭 상태로 표시 여부를 제어한다.

---

## 구현 방법

### 핵심 원리

Streamlit에서 `st.chat_input()`이 뷰포트 하단에 고정되려면
반드시 `st.tabs()` 컨텍스트 **밖**에 있어야 한다.

탭 UI는 유지하되, 채팅 입력창만 탭 블록 밖으로 빼낸다.
Q&A 탭이 활성화된 상태에서 vectorstore가 준비된 경우에만 입력창을 표시한다.

### 구조 변경

**변경 전:**
```
st.tabs([...])
  with tab_qa:
    ...히스토리 렌더링...
    st.chat_input(...)   ← 탭 안 → 버그 발생
```

**변경 후:**
```
st.tabs([...])
  with tab_qa:
    ...히스토리 렌더링...
    (입력창 없음)

# 탭 블록 완전히 종료 후
if "vectorstore" in st.session_state:
    st.chat_input(...)   ← 탭 밖 → 뷰포트 하단 고정
```

### 탭 전환 감지 (EDA 탭에서 입력창 숨김)

Streamlit은 탭 전환 이벤트를 직접 노출하지 않는다.
우회 방법: 각 탭 안에서 **버튼 클릭** 또는 **특정 위젯 interaction** 으로
`st.session_state["active_tab"]` 값을 갱신한다.

**구체적 구현**: `st.tabs()` 대신 **`st.radio()`로 탭을 직접 구현**한다.
이렇게 하면 선택 상태를 `session_state`에서 바로 읽을 수 있다.

```python
tab = st.radio("", ["대회 문서 Q&A", "자동 EDA 리포트"], horizontal=True, label_visibility="collapsed")
```

- `tab == "대회 문서 Q&A"` 일 때: Q&A UI + 탭 밖 `st.chat_input()` 표시
- `tab == "자동 EDA 리포트"` 일 때: EDA UI, 입력창 없음

### 전체 구조 (`app.py` 재작성)

`st.tabs()` 대신 `st.radio()`로 탭 전환을 구현한다.
`st.chat_input()`은 Q&A 탭이 선택되고 vectorstore가 준비된 경우에만 최상위 레벨에 표시한다.

```python
# 탭 선택 (st.tabs 대신 radio — chat_input 고정 문제 우회)
tab = st.radio("", ["대회 문서 Q&A", "자동 EDA 리포트"],
               horizontal=True, label_visibility="collapsed")
st.divider()

# ── Q&A 탭 ───────────────────────────────────────────────────────────────────
if tab == "대회 문서 Q&A":
    st.header("대회 문서 Q&A")
    # 파일 업로드 및 vectorstore 빌드 (기존 로직 유지)
    ...
    if "vectorstore" in st.session_state:
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.info("...")

# ── EDA 탭 ───────────────────────────────────────────────────────────────────
elif tab == "자동 EDA 리포트":
    st.header("자동 EDA 리포트")
    # CSV 업로드 및 EDA 실행 (기존 로직 유지)
    ...

# ── 채팅 입력창 (최상위 레벨 — 항상 뷰포트 하단 고정) ─────────────────────────
if tab == "대회 문서 Q&A" and "vectorstore" in st.session_state:
    if question := st.chat_input("대회 문서에 대해 질문하세요"):
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                from src.qa_chain import answer_question
                result = answer_question(st.session_state["vectorstore"], question)
            st.markdown(result["answer"])
            st.session_state["chat_history"].append({"role": "assistant", "content": result["answer"]})
            if result["sources"]:
                with st.expander("참고 문서"):
                    for i, doc in enumerate(result["sources"], 1):
                        src = doc.metadata.get("source", "unknown")
                        st.markdown(f"**[{i}] {src}**")
                        st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
```

`components.html` 스크롤 주입 코드는 제거한다
(최상위 레벨의 `st.chat_input()`은 Streamlit이 자동으로 하단 고정 + 자동 스크롤 처리).

---

## 수정 대상 파일

- **`app.py`** 전체 구조 재배치 (로직 변경 없음, 위치만 이동)

## 검증 방법

1. `streamlit run app.py`
2. PDF 업로드 후 탭 Q&A 확인
3. 입력창이 화면 **하단**에 고정되는지 확인
4. 질문 3개 연속 입력 시 최신 메시지가 입력창 바로 위에 위치하는지 확인
5. EDA 탭 전환 시 입력창 동작 확인
