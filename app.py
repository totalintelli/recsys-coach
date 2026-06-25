import os

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
                import tempfile

                from langchain_community.document_loaders import PyPDFLoader
                from src.qa_chain import build_vectorstore, warm_up_llm

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
                # 로컬 LLM을 지금 미리 로딩한다(첫 질문이 모델 로딩+추론을 한꺼번에 떠안아
                # WebSocket 타임아웃 나는 것을 방지). Upstage/OFFLINE이면 즉시 반환.
                warm_up_llm()
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

    def _run_eda_from_file(source) -> None:
        from src.eda import Checklist, generate_eda

        # 체크리스트 placeholder를 가장 먼저 만들어 즉시 화면에 진행 상황을 띄운다.
        st.subheader("진행 상황")
        checklist = Checklist(st.empty())
        info_placeholder = st.empty()

        # 파일 읽기도 체크리스트의 첫 항목으로 표시한다. 확장자로 CSV/Parquet 분기.
        # 업로드 자체의 전송률은 Streamlit이 노출하지 않으므로(파일이 다 올라온 뒤 실행됨),
        # 1GB read가 오래 걸리는 구간을 스피너 + 파일명·크기로 안내한다.
        name = getattr(source, "name", source)  # 업로더는 UploadedFile, 경로 입력은 str
        # getattr의 default는 항상 평가되므로 os.path.getsize를 직접 default로 쓰면
        # 업로더(path 아님)에서 TypeError. hasattr로 분기해 path일 때만 stat 호출.
        size_mb = (source.size if hasattr(source, "size") else os.path.getsize(source)) / 1024 / 1024
        checklist.start(f"데이터 파일 읽기 — {name} ({size_mb:.1f}MB)")
        with st.spinner(f"`{name}` 읽는 중… ({size_mb:.1f}MB)"):
            df = pd.read_parquet(source) if str(name).endswith(".parquet") else pd.read_csv(source)
        checklist.complete()
        info_placeholder.write(f"데이터 크기: **{df.shape[0]:,}행 × {df.shape[1]}열**")

        generate_eda(df, checklist)

    _UPLOAD_LIMIT_MB = 1024

    input_mode = st.radio(
        "입력 방식",
        ["파일 업로드 (1GB)", "서버 경로 직접 입력 (대용량)"],
        key="eda_input_mode",
    )

    if input_mode == "파일 업로드 (1GB)":
        data_file = st.file_uploader("CSV/Parquet 파일을 업로드하세요", type=["csv", "parquet"], key="eda_csv")
        if data_file:
            if data_file.size > _UPLOAD_LIMIT_MB * 1024 * 1024:
                st.error(
                    f"파일 크기가 {_UPLOAD_LIMIT_MB}MB를 초과했습니다 ({data_file.size / 1024 / 1024:.1f}MB). "
                    "대용량 파일은 '서버 경로 직접 입력'을 사용하세요."
                )
            else:
                _run_eda_from_file(data_file)
        else:
            st.info("CSV/Parquet 파일을 업로드하면 자동으로 EDA 리포트를 생성합니다.")
    else:
        path = st.text_input(
            "서버 내 CSV/Parquet 파일 경로를 입력하세요",
            placeholder="/data/train.parquet",
            key="eda_path",
        )
        if st.button("EDA 시작", key="eda_start") and path:
            if not path.endswith((".csv", ".parquet")):
                st.error("CSV(.csv) 또는 Parquet(.parquet) 파일만 지원합니다.")
            elif not os.path.isfile(path):
                st.error(f"파일을 찾을 수 없습니다: {path}")
            else:
                _run_eda_from_file(path)
        elif not path:
            st.info("파일 경로를 입력한 뒤 'EDA 시작' 버튼을 누르세요.")

# ── 채팅 입력창 (최상위 레벨 — 뷰포트 하단 고정) ──────────────────────────────
# st.tabs() 밖에 있어야 Streamlit이 하단 sticky로 고정한다.
# Q&A 탭이 선택되고 vectorstore가 준비된 경우에만 표시한다.
if tab == "대회 문서 Q&A" and "vectorstore" in st.session_state:
    if question := st.chat_input("대회 문서에 대해 질문하세요"):
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            from src.eda import Checklist
            from src.qa_chain import answer_question

            # 진행 상황을 EDA와 동일한 Checklist(단계별 소요 시간)로 상세 표시.
            # answer_question은 단일 progress(label) 콜백만 주므로, 새 단계가 오면
            # 직전 항목을 완료 처리하고 새 항목을 시작하는 어댑터로 잇는다.
            progress_box = st.empty()
            checklist = Checklist(progress_box)

            def _on_step(label: str) -> None:
                if checklist._items:  # 직전 단계 완료 처리
                    checklist.complete()
                checklist.start(label)

            with st.spinner("답변 생성 중..."):
                result = answer_question(st.session_state["vectorstore"], question, progress=_on_step)
                checklist.complete()  # 마지막 단계 완료
            progress_box.empty()  # 답변이 나오면 진행 표시는 정리

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
