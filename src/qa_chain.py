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
LLM_BACKEND        = os.getenv("LLM_BACKEND", "upstage").lower()  # "upstage" | "llama3" | "qwen"
HF_TOKEN           = os.getenv("HF_TOKEN", "")
LOCAL_MODEL_ID     = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
LLAMA_LOAD_IN_8BIT    = os.getenv("LLAMA_LOAD_IN_8BIT", "true").lower() == "true"
LLAMA_TEMPERATURE     = float(os.getenv("LLAMA_TEMPERATURE", "0.3"))
LLAMA_MAX_NEW_TOKENS  = int(os.getenv("LLAMA_MAX_NEW_TOKENS", "512"))

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

_LOCAL_LLM_TEMPLATE = (
    "<|im_start|>system\n"
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
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "{question}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# ── 노이즈 전처리 ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """문서 청크의 노이즈 문자를 정규화한다."""
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(?:\[object Object\][,\s]*)+", "", text)  # JS 객체 문자열화 산물 제거
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\S) {2,}", " ", text)  # 비공백 뒤의 중복 공백만 압축 (줄머리 들여쓰기 보존 → 중첩 목록 유지)
    return text.strip()


def _clean_answer(text: str) -> str:
    """LLM 출력의 잔여 노이즈를 제거한다."""
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\S) {2,}", " ", text)  # 비공백 뒤의 중복 공백만 압축 (줄머리 들여쓰기 보존 → 중첩 목록 유지)
    return text.strip()


def _md_to_html(text: str) -> str:
    """마크다운을 Jupyter Notebook 스타일 HTML로 변환한다.
    markdown-it-py + pygments로 코드 하이라이팅, 표, 목록 등을 완전히 렌더링한다.
    """
    if not text or not text.strip():
        return ""

    try:
        from markdown_it import MarkdownIt
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.formatters import HtmlFormatter
        from pygments.util import ClassNotFound
    except ImportError:
        # 폴백: markdown-it-py나 pygments가 없으면 기본 텍스트를 그대로 반환
        return text.replace("\n", "<br>")

    def _highlight_code(code: str, lang: str | None, attrs: str) -> str:
        if lang:
            try:
                lexer = get_lexer_by_name(lang.strip())
            except ClassNotFound:
                lexer = guess_lexer(code)
        else:
            lexer = guess_lexer(code)
        formatter = HtmlFormatter(
            style="default",
            noclasses=True,
            wrapcode=True,
            prestyles="margin:0; padding:0;",
        )
        return highlight(code, lexer, formatter)

    md = MarkdownIt("commonmark", {"highlight": _highlight_code})
    md.enable("table")

    html = md.render(text)

    # Jupyter Notebook 스타일 CSS
    css = """<style>
    .jupyter-markdown {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        line-height: 1.6;
        color: #ffffff;
    }
    .jupyter-markdown h1 {
        font-size: 2em;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.3em;
        margin-top: 1em;
        margin-bottom: 0.5em;
        color: #f0b429;
        font-weight: 600;
    }
    .jupyter-markdown h2 {
        font-size: 1.5em;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.3em;
        margin-top: 1em;
        margin-bottom: 0.5em;
        color: #f0b429;
        font-weight: 600;
    }
    .jupyter-markdown h3 {
        font-size: 1.25em;
        margin-top: 1em;
        margin-bottom: 0.5em;
        color: #f0b429;
        font-weight: 600;
    }
    .jupyter-markdown h4 {
        font-size: 1em;
        margin-top: 1em;
        margin-bottom: 0.5em;
        color: #f0b429;
        font-weight: 600;
    }
    .jupyter-markdown pre {
        background: #f7f7f7;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 16px;
        overflow: auto;
        font-size: 85%;
        line-height: 1.45;
        margin: 1em 0;
    }
    .jupyter-markdown code {
        background: rgba(27,31,35,0.05);
        border-radius: 3px;
        padding: 0.2em 0.4em;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 85%;
    }
    .jupyter-markdown pre code {
        background: transparent;
        padding: 0;
        border-radius: 0;
        font-size: 100%;
    }
    .jupyter-markdown blockquote {
        border-left: 4px solid #dfe2e5;
        padding-left: 16px;
        color: #6a737d;
        margin: 0 0 1em 0;
    }
    .jupyter-markdown table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
        overflow: auto;
        display: block;
    }
    .jupyter-markdown th, .jupyter-markdown td {
        border: 1px solid #dfe2e5;
        padding: 6px 13px;
    }
    .jupyter-markdown th {
        background: #f6f8fa;
        font-weight: 600;
    }
    .jupyter-markdown tr:nth-child(2n) {
        background: #f6f8fa;
    }
    .jupyter-markdown ul, .jupyter-markdown ol {
        padding-left: 2em;
        margin-bottom: 1em;
    }
    .jupyter-markdown li > ul, .jupyter-markdown li > ol {
        margin-bottom: 0;
    }
    .jupyter-markdown li {
        color: #ffffff;
    }
    .jupyter-markdown ul > li {
        list-style-type: disc;
    }
    .jupyter-markdown ul > li > ul > li {
        list-style-type: circle;
    }
    .jupyter-markdown img {
        max-width: 100%;
        box-sizing: border-box;
    }
    .jupyter-markdown a {
        color: #0366d6;
        text-decoration: none;
    }
    .jupyter-markdown a:hover {
        text-decoration: underline;
    }
    .jupyter-markdown hr {
        border: 0;
        border-top: 1px solid #e0e0e0;
        margin: 1em 0;
    }
    .jupyter-markdown p {
        margin-bottom: 1em;
    }
    </style>"""

    return f'<div class="jupyter-markdown">{html}</div>{css}'


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
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID, **kwargs)

    # ChatML 계열(<|im_end|>)과 eos를 모두 stop token으로 등록
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    terminators = list({tokenizer.eos_token_id, im_end_id} - {None})

    quant_config = None
    if LLAMA_LOAD_IN_8BIT and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        **kwargs,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=LLAMA_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=LLAMA_TEMPERATURE,
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=terminators,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        tokenizer_kwargs={"add_special_tokens": False},
    )
    _llama_llm_cache = HuggingFacePipeline(pipeline=pipe)
    return _llama_llm_cache


def _get_llama_prompt():
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate(input_variables=["context", "question"], template=_LOCAL_LLM_TEMPLATE)


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

    if LLM_BACKEND in ("llama3", "qwen"):
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
