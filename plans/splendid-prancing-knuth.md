# 답변 들여쓰기 계층 불릿 렌더링 수정

## Context

Streamlit의 `st.markdown()`은 마크다운 계층 불릿(`  - ` 들여쓰기)을 제대로 렌더링하지 못하고 평탄하게 출력한다. 이미지처럼 ●/○ 계층 불릿이 보이려면 마크다운을 HTML `<ul><li>` 구조로 변환한 뒤 `unsafe_allow_html=True`로 렌더링해야 한다.

LLM 출력은 마크다운 그대로 유지하고, Python 후처리에서 변환한다. 이 방식은 LLM이 들여쓰기를 조금 다르게 써도 안정적으로 동작한다.

## 변경 파일

**`src/qa_chain.py`** — `_clean_answer()` 하단 또는 별도 `_md_to_html()` 함수 추가.  
**`app.py`** — `st.markdown(result["answer"])` → `st.markdown(_md_to_html(result["answer"]), unsafe_allow_html=True)`로 변경 (115번 줄).

## 구체적 변경

### 1. `src/qa_chain.py` — `_md_to_html()` 추가

`_clean_answer()` 바로 아래에 다음 함수를 추가한다:

```python
def _md_to_html(text: str) -> str:
    """마크다운 계층 불릿·헤더를 HTML로 변환해 Streamlit 렌더링을 보정한다."""
    import re
    lines = text.split("\n")
    out = []
    list_depth = 0  # 현재 열려 있는 <ul> 깊이

    def close_lists(target_depth: int):
        nonlocal list_depth
        while list_depth > target_depth:
            out.append("</ul>")
            list_depth -= 1

    for line in lines:
        # ## 헤더
        h2 = re.match(r"^##\s+(.*)", line)
        if h2:
            close_lists(0)
            out.append(f"<h3 style='color:#5B5EA6;margin-top:0.8em'>{h2.group(1)}</h3>")
            continue

        # 들여쓰기 불릿 (- 또는 * 앞에 공백)
        bullet = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if bullet:
            indent = len(bullet.group(1))
            depth = (indent // 2) + 1  # 공백 2칸마다 깊이 1
            while list_depth < depth:
                out.append("<ul>")
                list_depth += 1
            close_lists(depth)
            out.append(f"<li>{bullet.group(2)}</li>")
            continue

        # 번호 목록
        num = re.match(r"^(\s*)\d+\.\s+(.*)", line)
        if num:
            close_lists(0)
            out.append(f"<li>{num.group(2)}</li>")
            continue

        # 빈 줄
        if line.strip() == "":
            close_lists(0)
            out.append("<br>")
            continue

        # 일반 텍스트
        close_lists(0)
        out.append(line)

    close_lists(0)
    return "\n".join(out)
```

이 함수는 `answer_question()`의 반환값에는 적용하지 않는다(캐시·저장에는 원본 마크다운 유지). 렌더링 시점에만 변환한다.

### 2. `app.py` — import 및 렌더링 변경

115번 줄:
```python
# 변경 전
st.markdown(result["answer"])

# 변경 후
from src.qa_chain import _md_to_html
st.markdown(_md_to_html(result["answer"]), unsafe_allow_html=True)
```

채팅 히스토리 재렌더링(64번 줄)도 같이 적용한다:
```python
# 변경 전 (64번 줄)
st.markdown(msg["content"])

# 변경 후 — assistant 메시지에만 적용
if msg["role"] == "assistant":
    from src.qa_chain import _md_to_html
    st.markdown(_md_to_html(msg["content"]), unsafe_allow_html=True)
else:
    st.markdown(msg["content"])
```

## 검증

`streamlit run app.py` 후 계층 불릿이 포함된 답변(예: "Train-Test Split 설명해줘")을 질문했을 때 ●/○ 계층이 시각적으로 구분되어 렌더링되는지 확인. `##` 헤더가 보라색으로 표시되는지도 확인.
