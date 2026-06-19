import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

st.set_page_config(page_title="RecSys Coach", page_icon="🏆", layout="wide")
st.title("🏆 RecSys Coach")
st.caption("추천 시스템 대회 참가자를 위한 AI 코칭 도우미")

with st.sidebar:
    st.header("메뉴")
    tab = st.radio(
        "탭 선택",
        ["대회 문서 Q&A", "자동 EDA 리포트"],
        label_visibility="collapsed",
    )

# ── Q&A 탭 ───────────────────────────────────────────────────────────────────
if tab == "대회 문서 Q&A":
    st.header("대회 문서 Q&A")

    uploaded_files = st.file_uploader(
        "대회 문서를 업로드하세요 (PDF, TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if "vectorstore" not in st.session_state or st.session_state.get("qa_files") != [f.name for f in uploaded_files]:
            with st.spinner("문서를 분석 중입니다..."):
                import os
                import tempfile

                from langchain_community.document_loaders import PyPDFLoader
                from src.qa_chain import build_vectorstore

                docs: list[Document] = []
                for uf in uploaded_files:
                    raw = uf.read()
                    if uf.name.endswith(".pdf"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(raw)
                            tmp_path = tmp.name
                        loader = PyPDFLoader(tmp_path)
                        loaded = loader.load()
                        for doc in loaded:
                            doc.metadata["source"] = uf.name
                        docs.extend(loaded)
                        os.unlink(tmp_path)
                    else:
                        text = raw.decode("utf-8", errors="ignore")
                        docs.append(Document(page_content=text, metadata={"source": uf.name}))

                st.session_state["vectorstore"] = build_vectorstore(docs)
                st.session_state["qa_files"] = [f.name for f in uploaded_files]
                st.session_state.setdefault("chat_history", [])
            st.success(f"{len(uploaded_files)}개 파일 인덱싱 완료")

    if "vectorstore" in st.session_state:
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    from src.qa_chain import _md_to_html
                    st.markdown(_md_to_html(msg["content"]), unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])
    else:
        st.info("위 업로더에서 대회 문서(PDF 또는 TXT)를 업로드하면 Q&A를 시작할 수 있습니다.")

# ── EDA 탭 ───────────────────────────────────────────────────────────────────
elif tab == "자동 EDA 리포트":
    st.header("자동 EDA 리포트")

    csv_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"], key="eda_csv")

    if csv_file:
        df = pd.read_csv(csv_file)
        st.write(f"데이터 크기: **{df.shape[0]:,}행 × {df.shape[1]}열**")

        with st.spinner("EDA 리포트를 생성 중입니다..."):
            from src.eda import generate_eda
            generate_eda(df)
    else:
        st.info("CSV 파일을 업로드하면 자동으로 EDA 리포트를 생성합니다.")

# ── 채팅 입력창 (최상위 레벨 — 뷰포트 하단 고정) ──────────────────────────────
# st.tabs() 밖에 있어야 Streamlit이 하단 sticky로 고정한다.
# Q&A 탭이 선택되고 vectorstore가 준비된 경우에만 표시한다.
if tab == "대회 문서 Q&A" and "vectorstore" in st.session_state:
    if question := st.chat_input("대회 문서에 대해 질문하세요"):
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                from src.qa_chain import answer_question
                result = answer_question(st.session_state["vectorstore"], question)

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
                    st.markdown("")

            from src.qa_chain import _md_to_html
            st.markdown(_md_to_html(result["answer"]), unsafe_allow_html=True)

            st.session_state["chat_history"].append({"role": "assistant", "content": result["answer"]})

            if result["sources"]:
                with st.expander("참고 문서"):
                    seen_src = set()
                    for doc in result["sources"]:
                        src = doc.metadata.get("source", "unknown")
                        if src not in seen_src:
                            seen_src.add(src)
                            st.markdown(f"- {src}")
