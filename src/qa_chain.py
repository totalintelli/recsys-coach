import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

OFFLINE_MODE       = os.getenv("OFFLINE_MODE", "false").lower() == "true"
UPSTAGE_API_KEY    = os.getenv("UPSTAGE_API_KEY", "")
LLM_BACKEND        = os.getenv("LLM_BACKEND", "upstage").lower()  # "upstage" | "llama3"
HF_TOKEN           = os.getenv("HF_TOKEN", "")
LLAMA_MODEL_ID     = os.getenv("LLAMA_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
LLAMA_LOAD_IN_8BIT = os.getenv("LLAMA_LOAD_IN_8BIT", "true").lower() == "true"

_llama_llm_cache: Any | None = None
_answer_cache: dict[tuple[int, str], dict] = {}

# ── 프롬프트 ──────────────────────────────────────────────────────────────────

_UPSTAGE_SYSTEM_PROMPT = """\
당신은 추천 시스템 대회 전문 AI 코치입니다.
아래 참고 문서를 바탕으로 질문에 정확하고 구체적으로 답변하세요.

규칙:
- 문서에 명시된 내용을 우선하고, 문서에 없는 내용은 "문서에서 확인할 수 없습니다"라고 답변하세요.
- 코드나 수식이 포함된 경우 마크다운 코드 블록(```python)을 사용하세요.
- 답변의 핵심 주제가 있으면 ## 헤더로 제목을 붙이세요.
- 목록이나 단계가 있는 경우 번호 목록 또는 불릿 목록으로 정리하고, 하위 항목은 들여쓰기 불릿(  - )으로 계층을 표현하세요.
- 답변은 한국어로, 명확하고 간결하게 작성하세요.
- 날짜나 기간이 여러 개 나열될 때는 반드시 과거에서 현재 순서(오름차순)로 배치하세요.

참고 문서:
{context}"""

_LLAMA3_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "당신은 추천 시스템 대회 전문 AI 코치입니다.\n"
    "아래 참고 문서를 바탕으로 질문에 정확하고 구체적으로 답변하세요.\n\n"
    "규칙:\n"
    "- 문서에 명시된 내용을 우선하고, 문서에 없는 내용은 '문서에서 확인할 수 없습니다'라고 답변하세요.\n"
    "- 코드나 수식이 포함된 경우 마크다운 코드 블록을 사용하세요.\n"
    "- 답변의 핵심 주제가 있으면 ## 헤더로 제목을 붙이세요.\n"
    "- 목록이나 단계가 있는 경우 불릿 목록으로 정리하고, 하위 항목은 들여쓰기 불릿(  - )으로 계층을 표현하세요.\n"
    "- 답변은 한국어로 작성하세요.\n"
    "- 날짜나 기간이 여러 개 나열될 때는 반드시 과거에서 현재 순서(오름차순)로 배치하세요.\n\n"
    "참고 문서:\n{context}"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{question}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# ── 노이즈 전처리 ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """문서 청크의 노이즈 문자를 정규화한다."""
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _clean_answer(text: str) -> str:
    """LLM 출력의 잔여 노이즈를 제거한다."""
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _md_to_html(text: str) -> str:
    """마크다운 계층 불릿·헤더를 HTML로 변환해 Streamlit 렌더링을 보정한다."""
    lines = text.split("\n")
    out: list[str] = []
    list_depth = 0

    def close_lists(target_depth: int) -> None:
        nonlocal list_depth
        while list_depth > target_depth:
            out.append("</ul>")
            list_depth -= 1

    for line in lines:
        h2 = re.match(r"^##\s+(.*)", line)
        if h2:
            close_lists(0)
            out.append(f"<h3 style='color:#5B5EA6;margin-top:0.8em'>{h2.group(1)}</h3>")
            continue

        bullet = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if bullet:
            indent = len(bullet.group(1))
            depth = (indent // 2) + 1
            while list_depth < depth:
                out.append("<ul>")
                list_depth += 1
            close_lists(depth)
            out.append(f"<li>{bullet.group(2)}</li>")
            continue

        num = re.match(r"^(\s*)\d+\.\s+(.*)", line)
        if num:
            close_lists(0)
            out.append(f"<li>{num.group(2)}</li>")
            continue

        if line.strip() == "":
            close_lists(0)
            out.append("<br>")
            continue

        close_lists(0)
        out.append(line)

    close_lists(0)
    return "\n".join(out)


def _enrich_chunk(doc: Document) -> Document:
    """청크에 섹션 출처 헤더를 prefix로 추가한다."""
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page")
    header = f"[출처: {source}" + (f", {page + 1}페이지" if page is not None else "") + "]\n"
    cleaned = _clean_text(doc.page_content)
    return Document(page_content=header + cleaned, metadata=doc.metadata)


# ── 청킹 ─────────────────────────────────────────────────────────────────────

def _split_documents(docs: list[Document]) -> list[Document]:
    """문단/문장 경계 기반으로 문서를 청크로 분리한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
        keep_separator=False,
    )
    return splitter.split_documents(docs)


# ── Reranker ──────────────────────────────────────────────────────────────────

def _rerank_with_cross_encoder(query: str, docs: list[Document], top_k: int = 3) -> list[Document]:
    """Cross-Encoder로 문서를 재순위한다. sentence-transformers 미설치 시 단순 슬라이스 fallback."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = model.predict([(query, doc.page_content) for doc in docs])
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
    except ImportError:
        return docs[:top_k]
    except Exception:
        import warnings
        warnings.warn("[recsys-coach] Reranker 실패, 단순 슬라이스로 폴백합니다.")
        return docs[:top_k]


# ── LLM / 프롬프트 헬퍼 ──────────────────────────────────────────────────────

def _get_embeddings():
    from langchain_upstage import UpstageEmbeddings
    return UpstageEmbeddings(
        api_key=UPSTAGE_API_KEY,
        model="solar-embedding-1-large",
    )


def _get_upstage_llm():
    from langchain_upstage import ChatUpstage
    return ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")


def _get_upstage_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_messages([
        ("system", _UPSTAGE_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])


def _get_llama_llm():
    global _llama_llm_cache
    if _llama_llm_cache is not None:
        return _llama_llm_cache

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    from langchain_huggingface import HuggingFacePipeline

    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, **kwargs)

    quant_config = None
    if LLAMA_LOAD_IN_8BIT and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        **kwargs,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    _llama_llm_cache = HuggingFacePipeline(pipeline=pipe)
    return _llama_llm_cache


def _get_llama_prompt():
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate(input_variables=["context", "question"], template=_LLAMA3_TEMPLATE)


# ── 하이브리드 검색 ───────────────────────────────────────────────────────────

def _build_retriever(vectorstore: FAISS, chunks: list[Document]):
    """BM25 + FAISS 앙상블 리트리버를 반환한다. rank-bm25 미설치 시 FAISS만 사용."""
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        bm25 = BM25Retriever.from_documents(chunks, k=6)
        return EnsembleRetriever(
            retrievers=[bm25, faiss_retriever],
            weights=[0.4, 0.6],
        )
    except ImportError:
        return faiss_retriever


# ── 공개 API ─────────────────────────────────────────────────────────────────

def build_vectorstore(docs: list[Document]) -> FAISS:
    chunks = _split_documents(docs)
    enriched = [_enrich_chunk(c) for c in chunks]
    vs = FAISS.from_documents(enriched, _get_embeddings())
    vs._chunks = enriched  # type: ignore[attr-defined]
    return vs


def answer_question(vectorstore: FAISS, question: str) -> dict[str, Any]:
    """{'answer': str, 'sources': list[Document]} 반환"""
    # ── 캐시 확인 ────────────────────────────────────────────────────────────
    cache_key = (id(vectorstore), question)
    if cache_key in _answer_cache:
        return _answer_cache[cache_key]

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    chunks = getattr(vectorstore, "_chunks", None) or []
    retriever = _build_retriever(vectorstore, chunks)
    source_docs = retriever.invoke(question)

    # Cross-Encoder reranking → 상위 3개
    context_docs = _rerank_with_cross_encoder(question, source_docs, top_k=3)

    if OFFLINE_MODE:
        answer = _clean_answer("\n\n---\n\n".join(doc.page_content for doc in context_docs))
        result = {"answer": answer, "sources": context_docs}
        _answer_cache[cache_key] = result
        return result

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _run_chain(llm, prompt) -> str:
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain.invoke(question)

    if LLM_BACKEND == "llama3":
        raw_answer = _run_chain(_get_llama_llm(), _get_llama_prompt())
        result = {"answer": _clean_answer(raw_answer), "sources": context_docs}
        _answer_cache[cache_key] = result
        return result

    try:
        raw_answer = _run_chain(_get_upstage_llm(), _get_upstage_prompt())
    except Exception as exc:
        import warnings
        warnings.warn(f"[recsys-coach] Upstage 호출 실패 ({exc!r}), Llama-3으로 폴백합니다.")
        raw_answer = _run_chain(_get_llama_llm(), _get_llama_prompt())

    result = {"answer": _clean_answer(raw_answer), "sources": context_docs}
    _answer_cache[cache_key] = result
    return result
