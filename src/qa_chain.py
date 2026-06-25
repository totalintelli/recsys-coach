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
LLM_BACKEND        = os.getenv("LLM_BACKEND", "upstage").lower()  # "upstage" | "llama3" | "qwen" | "vllm"
# 같은 가중치를 로컬에서 추론하는 백엔드 집합. 판정을 한 곳으로 모아 신규 백엔드(vllm) 추가 시
# 호출부 3곳이 따로 깨지지 않게 한다. "vllm"은 greedy면 transformers와 출력이 동치이면서 2~4배 빠름.
_LOCAL_BACKENDS    = {"llama3", "qwen", "vllm"}
HF_TOKEN           = os.getenv("HF_TOKEN", "")
LOCAL_MODEL_ID     = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
# 기본 fp16: 8bit는 dequant 오버헤드로 추론이 느리다. Qwen 3B fp16≈6GB라 24GB GPU에 여유.
# VRAM이 빠듯하면 LLAMA_LOAD_IN_8BIT=true로 8bit 복귀.
LLAMA_LOAD_IN_8BIT    = os.getenv("LLAMA_LOAD_IN_8BIT", "false").lower() == "true"
LLAMA_TEMPERATURE     = float(os.getenv("LLAMA_TEMPERATURE", "0"))  # 0=greedy(QA 권장·더 빠름), >0=sampling
LLAMA_MAX_NEW_TOKENS  = int(os.getenv("LLAMA_MAX_NEW_TOKENS", "256"))
RETRIEVER_K           = int(os.getenv("RETRIEVER_K", "6"))
LOCAL_RETRIEVER_K     = int(os.getenv("LOCAL_RETRIEVER_K", "4"))
RERANK_TOP_K          = int(os.getenv("RERANK_TOP_K", "3"))
LOCAL_RERANK_TOP_K    = int(os.getenv("LOCAL_RERANK_TOP_K", "2"))
RERANKER_ENABLED      = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_DEVICE       = os.getenv("RERANKER_DEVICE", "").strip()
LOCAL_DEVICE          = os.getenv("LOCAL_DEVICE", "").strip()
LOCAL_ATTN_IMPLEMENTATION = os.getenv("LOCAL_ATTN_IMPLEMENTATION", "sdpa").strip()
# vLLM 전용. GPU 메모리 점유율(0~1)·최대 컨텍스트 길이. 3090(24GB)+7B fp16 기준 기본값으로 충분.
VLLM_GPU_MEM_UTIL     = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.90"))
VLLM_MAX_MODEL_LEN    = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))

_llama_llm_cache: Any | None = None
_reranker_cache: Any | None = None  # CrossEncoder는 질문마다 재로딩하면 느려 모듈 캐시
_retriever_cache: dict[tuple[int, int], Any] = {}
_answer_cache: dict[tuple[int, str], dict] = {}

# ── 프롬프트 ──────────────────────────────────────────────────────────────────

_UPSTAGE_SYSTEM_PROMPT = """\
당신은 추천 시스템 대회 전문 AI 코치입니다.
아래 참고 문서를 바탕으로 질문에 정확하고 구체적으로 답변하세요.

규칙:
- 문서에 명시된 내용을 우선하고, 문서에 없는 내용은 "문서에서 확인할 수 없습니다"라고 답변하세요.
- 코드나 수식이 포함된 경우 마크다운 코드 블록(```python)을 사용하세요.
- 답변의 핵심 주제가 있으면 ## 헤더로 제목을 붙이세요.
- 본문 내용이 없는 제목이나 섹션은 만들지 마세요.
- 목록이나 단계가 있는 경우 번호 목록 또는 불릿 목록으로 정리하고, 하위 항목은 들여쓰기 불릿(  - )으로 계층을 표현하세요.
- 답변은 한국어로, 명확하고 간결하게 작성하세요.
- 답변 본문은 반드시 한국어만 사용하고, 중국어·일본어 표현을 섞지 마세요.
- 코드, 파일명, 함수명, 컬럼명, 모델명, API명 같은 고유 식별자는 원문 표기를 유지할 수 있습니다.
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
    "- 본문 내용이 없는 제목이나 섹션은 만들지 마세요.\n"
    "- 목록이나 단계가 있는 경우 불릿 목록으로 정리하고, 하위 항목은 들여쓰기 불릿(  - )으로 계층을 표현하세요.\n"
    "- 답변 본문은 반드시 한국어만 사용하세요.\n"
    "- 중국어/일본어 문장, 단어, 한자, 가나를 사용하지 마세요.\n"
    "- 코드, 파일명, 함수명, 컬럼명, 모델명, API명 같은 고유 식별자는 원문 표기를 유지할 수 있습니다.\n"
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
    # LLM이 컨텍스트의 JS 객체 문자열화 산물([object Object])을 따라 출력하는 경우 제거.
    # 모든 답변이 이 함수를 거치므로 출력 경계에서 한 번에 차단(_clean_text와 동일 규칙).
    text = re.sub(r"(?:\[object Object\][,\s]*)+", "", text)
    # [object Object] 제거 후 내용이 비어버린 코드블록(빈 박스)을 삭제 — 화면에 빈 상자만 남는 것 방지.
    text = re.sub(r"```[^\n]*\n\s*```", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\S) {2,}", " ", text)  # 비공백 뒤의 중복 공백만 압축 (줄머리 들여쓰기 보존 → 중첩 목록 유지)
    text = _remove_trailing_empty_headings(text)
    return text.strip()


def _remove_trailing_empty_headings(text: str) -> str:
    """답변 끝에 본문 없이 남은 마크다운 제목을 제거한다."""
    lines = text.rstrip().splitlines()
    heading_pattern = re.compile(r"^\s{0,3}#{1,6}\s+\S.*$")

    while lines and heading_pattern.match(lines[-1]):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()

    return "\n".join(lines)


def _contains_disallowed_cjk(text: str) -> bool:
    """코드 영역을 제외한 답변 본문에 중국어/일본어 문자가 섞였는지 확인한다."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", "", text)
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))


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
    global _reranker_cache
    if not RERANKER_ENABLED:
        return docs[:top_k]
    try:
        if _reranker_cache is None:  # 첫 호출 1회만 로딩, 이후 재사용
            from sentence_transformers import CrossEncoder
            reranker_kwargs = {"device": RERANKER_DEVICE} if RERANKER_DEVICE else {}
            _reranker_cache = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                **reranker_kwargs,
            )
        model = _reranker_cache
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


def _get_vllm_llm():
    """vLLM 백엔드. PagedAttention + continuous batching으로 같은 7B fp16을 transformers보다
    2~4배 빠르게 디코딩한다. greedy(temperature=0)면 출력이 transformers 경로와 동치이므로
    품질 손실 없이 속도만 올린다. langchain_community.llms.VLLM 래퍼 사용."""
    from langchain_community.llms import VLLM

    # greedy면 temperature=0(vLLM에서 그대로 greedy). sampling이면 transformers 경로와 같은 값 매핑.
    sampling = {"temperature": LLAMA_TEMPERATURE, "top_p": 0.9} if LLAMA_TEMPERATURE > 0 \
        else {"temperature": 0.0}
    # repetition_penalty는 샘플링 인자라 래퍼 최상위 필드로 받는다. 엔진 인자(gpu_memory_utilization,
    # max_model_len)는 vllm_kwargs로만 EngineArgs에 전달된다. 둘을 섞으면 EngineArgs가
    # repetition_penalty를 모른다며 TypeError를 낸다.
    return VLLM(
        model=LOCAL_MODEL_ID,
        dtype="float16",
        trust_remote_code=True,
        max_new_tokens=LLAMA_MAX_NEW_TOKENS,
        repetition_penalty=1.15,
        # ChatML stop. _LOCAL_LLM_TEMPLATE가 <|im_end|>로 턴을 닫으므로 거기서 멈춘다.
        stop=["<|im_end|>"],
        vllm_kwargs={
            "gpu_memory_utilization": VLLM_GPU_MEM_UTIL,
            "max_model_len": VLLM_MAX_MODEL_LEN,
        },
        **sampling,
    )


def _get_llama_llm():
    global _llama_llm_cache
    if _llama_llm_cache is not None:
        return _llama_llm_cache

    if LLM_BACKEND == "vllm":
        _llama_llm_cache = _get_vllm_llm()
        return _llama_llm_cache

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    from langchain_huggingface import HuggingFacePipeline

    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID, **kwargs)

    # ChatML 계열(<|im_end|>)과 eos를 모두 stop token으로 등록
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    terminators = list({tokenizer.eos_token_id, im_end_id} - {None})

    has_cuda = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    has_mps = bool(mps_backend and mps_backend.is_available())

    if has_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    quant_config = None
    if LLAMA_LOAD_IN_8BIT and has_cuda:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    torch_dtype = torch.float16 if has_cuda or has_mps else torch.float32
    default_device = "cuda:0" if has_cuda else "mps" if has_mps else "cpu"
    local_device = LOCAL_DEVICE or default_device
    device_map: Any = "auto" if local_device.lower() == "auto" else {"": local_device}

    model_kwargs = dict(
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
    )
    if LOCAL_ATTN_IMPLEMENTATION:
        model_kwargs["attn_implementation"] = LOCAL_ATTN_IMPLEMENTATION

    try:
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_ID,
            **model_kwargs,
            **kwargs,
        )
    except (TypeError, ValueError, ImportError):
        if "attn_implementation" not in model_kwargs:
            raise
        model_kwargs.pop("attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_ID,
            **model_kwargs,
            **kwargs,
        )

    # QA(사실 추출)는 greedy가 sampling보다 빠르고 결정적이다. temp<=0이면 greedy로,
    # temp>0이면 기존 sampling 동작 유지(LLAMA_TEMPERATURE로 토글). greedy 시 top_p/temperature는
    # 무의미하므로 generate에 넘기지 않는다(경고 방지).
    gen_kwargs = dict(
        max_new_tokens=LLAMA_MAX_NEW_TOKENS,
        repetition_penalty=1.15,
        eos_token_id=terminators,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if LLAMA_TEMPERATURE > 0:
        gen_kwargs.update(do_sample=True, temperature=LLAMA_TEMPERATURE, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **gen_kwargs)
    _llama_llm_cache = HuggingFacePipeline(pipeline=pipe)
    return _llama_llm_cache


def _get_llama_prompt():
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate(input_variables=["context", "question"], template=_LOCAL_LLM_TEMPLATE)


# ── 하이브리드 검색 ───────────────────────────────────────────────────────────

def _build_retriever(vectorstore: FAISS, chunks: list[Document], k: int = RETRIEVER_K):
    """BM25 + FAISS 앙상블 리트리버를 반환한다. rank-bm25 미설치 시 FAISS만 사용."""
    cache_key = (id(vectorstore), k)
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        bm25 = BM25Retriever.from_documents(chunks, k=k)
        retriever = EnsembleRetriever(
            retrievers=[bm25, faiss_retriever],
            weights=[0.4, 0.6],
        )
    except ImportError:
        retriever = faiss_retriever

    _retriever_cache[cache_key] = retriever
    return retriever


# ── 공개 API ─────────────────────────────────────────────────────────────────

def build_vectorstore(docs: list[Document]) -> FAISS:
    chunks = _split_documents(docs)
    enriched = [_enrich_chunk(c) for c in chunks]
    vs = FAISS.from_documents(enriched, _get_embeddings())
    vs._chunks = enriched  # type: ignore[attr-defined]
    return vs


def warm_up_llm() -> None:
    """로컬 LLM(LLaMA/Qwen)을 미리 로딩해 캐시에 채운다.

    첫 질문이 '모델 로딩 + 추론'을 한꺼번에 떠안아 WebSocket 타임아웃이 나는 것을 막는다.
    문서 인덱싱 직후(이미 spinner가 도는 구간)에 호출하면 실제 질문은 추론만 한다.
    Upstage(API)·OFFLINE 모드는 로딩이 없으므로 아무 것도 하지 않는다.
    """
    # 리랭커도 첫 질문에서 분리(질문마다 재로딩하던 것을 미리 캐시).
    _rerank_with_cross_encoder("warm up", [Document(page_content="warm up")], top_k=1)
    if not OFFLINE_MODE and LLM_BACKEND in _LOCAL_BACKENDS:
        _get_llama_llm()  # 캐시(_llama_llm_cache)를 채운다


def answer_question(vectorstore: FAISS, question: str, progress=None) -> dict[str, Any]:
    """{'answer': str, 'sources': list[Document]} 반환.

    progress: 선택적 콜백 (label: str) -> None. 각 단계 시작 시 호출돼 진행 상황을
    UI에 표시한다. None이면(기본) 아무 것도 하지 않아 기존 호출부와 호환된다.
    """
    def _step(label: str) -> None:
        if progress is not None:
            progress(label)

    # ── 캐시 확인 ────────────────────────────────────────────────────────────
    cache_key = (id(vectorstore), question)
    if cache_key in _answer_cache:
        _step("캐시된 답변 사용")
        return _answer_cache[cache_key]

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    is_local_backend = LLM_BACKEND in _LOCAL_BACKENDS
    retriever_k = LOCAL_RETRIEVER_K if is_local_backend else RETRIEVER_K
    rerank_top_k = LOCAL_RERANK_TOP_K if is_local_backend else RERANK_TOP_K

    _step("문서 검색 (BM25 + FAISS)")
    chunks = getattr(vectorstore, "_chunks", None) or []
    retriever = _build_retriever(vectorstore, chunks, k=retriever_k)
    source_docs = retriever.invoke(question)

    # Cross-Encoder reranking. 로컬 LLM은 기본 컨텍스트를 줄여 답변 생성 시간을 낮춘다.
    if RERANKER_ENABLED:
        _step("관련도 재정렬 (Cross-Encoder)")
    context_docs = _rerank_with_cross_encoder(question, source_docs, top_k=rerank_top_k)

    if OFFLINE_MODE:
        _step("오프라인 모드 — 검색 컨텍스트 반환")
        answer = _clean_answer("\n\n---\n\n".join(doc.page_content for doc in context_docs))
        result = {"answer": answer, "sources": context_docs}
        _answer_cache[cache_key] = result
        return result

    # 리트리버는 위에서 이미 1회 실행해 리랭킹까지 마쳤다(context_docs).
    # 체인 안에서 retriever를 다시 호출하면 앙상블 검색이 질문마다 2회 돌고,
    # LLM에는 리랭킹 안 된 원본이 가는 불일치도 생긴다. 리랭킹된 컨텍스트를 그대로 주입.
    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

    def _run_chain(llm, prompt, input_question: str = question) -> str:
        chain = (
            {"context": lambda _: context_text, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain.invoke(input_question)

    if is_local_backend:
        _step(f"답변 생성 ({LLM_BACKEND})")
        llm = _get_llama_llm()
        prompt = _get_llama_prompt()
        raw_answer = _run_chain(llm, prompt)
        answer = _clean_answer(raw_answer)
        if _contains_disallowed_cjk(answer):
            _step("중국어/일본어 혼입 감지 — 한국어 답변 재생성")
            retry_question = (
                f"{question}\n\n"
                "이전 답변에 중국어 또는 일본어가 섞였습니다. "
                "답변 본문은 반드시 한국어로만 다시 작성하세요. "
                "중국어/일본어 문장, 단어, 한자, 가나는 사용하지 마세요. "
                "코드, 파일명, 함수명, 컬럼명, 모델명, API명 같은 고유 식별자는 원문 표기를 유지할 수 있습니다."
            )
            answer = _clean_answer(_run_chain(llm, prompt, input_question=retry_question))
        result = {"answer": answer, "sources": context_docs}
        _answer_cache[cache_key] = result
        return result

    try:
        _step("답변 생성 (Solar-Pro)")
        raw_answer = _run_chain(_get_upstage_llm(), _get_upstage_prompt())
    except Exception as exc:
        import warnings
        warnings.warn(f"[recsys-coach] Upstage 호출 실패 ({exc!r}), Llama-3으로 폴백합니다.")
        _step("Solar 실패 — Llama-3 폴백 생성")
        raw_answer = _run_chain(_get_llama_llm(), _get_llama_prompt())

    result = {"answer": _clean_answer(raw_answer), "sources": context_docs}
    _answer_cache[cache_key] = result
    return result
