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

# ── 프롬프트 ──────────────────────────────────────────────────────────────────

_UPSTAGE_SYSTEM_PROMPT = """\
당신은 추천 시스템 대회 전문 AI 코치입니다.
아래 참고 문서를 바탕으로 질문에 정확하고 구체적으로 답변하세요.

규칙:
- 문서에 명시된 내용을 우선하고, 문서에 없는 내용은 "문서에서 확인할 수 없습니다"라고 답변하세요.
- 코드나 수식이 포함된 경우 마크다운 코드 블록(```python)을 사용하세요.
- 목록이나 단계가 있는 경우 번호 목록 또는 불릿 목록으로 정리하세요.
- 답변은 한국어로, 명확하고 간결하게 작성하세요.

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
    "- 답변은 한국어로 작성하세요.\n\n"
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
    # 이스케이프된 개행 복원
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    # HTML 태그 제거 (<br>, <br/>, <p>, 등)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    # 연속 공백/개행 정리 (3개 이상 → 2개)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _enrich_chunk(doc: Document) -> Document:
    """청크에 섹션 출처 헤더를 prefix로 추가한다."""
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page")
    header = f"[출처: {source}" + (f", {page + 1}페이지" if page is not None else "") + "]\n"
    cleaned = _clean_text(doc.page_content)
    return Document(page_content=header + cleaned, metadata=doc.metadata)


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
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        bm25 = BM25Retriever.from_documents(chunks, k=5)
        return EnsembleRetriever(
            retrievers=[bm25, faiss_retriever],
            weights=[0.4, 0.6],
        )
    except ImportError:
        return faiss_retriever


# ── 공개 API ─────────────────────────────────────────────────────────────────

def build_vectorstore(docs: list[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    enriched = [_enrich_chunk(c) for c in chunks]

    vs = FAISS.from_documents(enriched, _get_embeddings())
    # 하이브리드 검색용으로 청크 목록도 vectorstore 객체에 보관
    vs._chunks = enriched  # type: ignore[attr-defined]
    return vs


def answer_question(vectorstore: FAISS, question: str) -> dict[str, Any]:
    """{'answer': str, 'sources': list[Document]} 반환"""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    chunks = getattr(vectorstore, "_chunks", None) or []
    retriever = _build_retriever(vectorstore, chunks)

    source_docs = retriever.invoke(question)
    # 상위 3개만 LLM 컨텍스트에 전달 (앙상블은 k=5로 가져왔으므로 재절삭)
    context_docs = source_docs[:3]

    if OFFLINE_MODE:
        answer = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
        return {"answer": answer, "sources": context_docs}

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
        answer = _run_chain(_get_llama_llm(), _get_llama_prompt())
        return {"answer": answer, "sources": source_docs}

    try:
        answer = _run_chain(_get_upstage_llm(), _get_upstage_prompt())
    except Exception as exc:
        import warnings
        warnings.warn(f"[recsys-coach] Upstage 호출 실패 ({exc!r}), Llama-3으로 폴백합니다.")
        answer = _run_chain(_get_llama_llm(), _get_llama_prompt())

    return {"answer": answer, "sources": source_docs}
