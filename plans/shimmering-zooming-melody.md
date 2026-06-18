# 출처 표시 방식 전환: LLM 인라인 태그 → Python 후처리

## Context

LLM(solar-pro)이 context 태그를 무시하고 문서 내용에서 출처 제목을 자체적으로 만들어낸다(`[리더보드 일일 줄헐확수, 1 page]` 등). 프롬프트로 강제하는 방식은 신뢰할 수 없으므로, LLM은 답변 텍스트만 생성하고 Python이 `source_docs` 메타데이터를 직접 읽어 답변 끝에 출처 목록을 붙이는 방식으로 전환한다.

## 수정 대상

### 1. `src/qa_chain.py`

**시스템 프롬프트** — 출처 태그 지시 제거, 순수 답변만 생성하도록 단순화

```python
# 변경 후
prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 문서를 참고하여 질문에 답하세요.\n\n{context}"),
    ("human", "{question}"),
])
```

**format_docs()** — 태그 없이 내용만 전달 (LLM이 출처를 판단할 근거 제거)

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

### 2. `app.py` 라인 98-108

답변 텍스트 뒤에 Python이 `source_docs`를 순회해 `[파일명, N page]` 목록을 조합해 붙인다. 중복 제거는 `(src, page)` 튜플 기준.

```python
st.markdown(result["answer"])

if result["sources"]:
    seen = set()
    tags = []
    for doc in result["sources"]:
        src = doc.metadata.get("source", "")
        page = doc.metadata.get("page")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            if src and page is not None:
                tags.append(f"[{src}, {page + 1} page]")
            elif src:
                tags.append(f"[{src}]")
    if tags:
        st.caption(" · ".join(tags))

st.session_state["chat_history"].append({"role": "assistant", "content": result["answer"]})

if result["sources"]:
    with st.expander("참고 문서"):
        seen_src = set()
        for doc in result["sources"]:
            src = doc.metadata.get("source", "unknown")
            if src not in seen_src:
                seen_src.add(src)
                st.markdown(f"- {src}")
```

`st.caption()`으로 답변 바로 아래 회색 소자로 출처 목록을 표시한다. `chat_history`에는 답변 텍스트만 저장(출처 태그 제외).

## 결과 예시

```
DialogSum 데이터셋을 기반으로 한 모델/파싱 데이터는 사용 금지됩니다.

일상_대화_요약_데이터_개요.pdf, 1 page · 일상_대화_요약_규정.pdf, 2 page
```

## 검증

1. `streamlit run app.py` 실행
2. PDF 2개 이상 업로드 후 질문
3. 답변 텍스트에 임의 출처 태그가 없는지 확인
4. 답변 아래 `st.caption()`으로 실제 파일명·페이지가 표시되는지 확인
