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
# Upstage Solar 모델 별칭. solar-pro(1세대) | solar-pro2 | solar-pro3. 선택 안 넘어왔을 때 폴백.
# 별칭별로 가리키는 모델이 다르다(자동 최신 아님).
UPSTAGE_MODEL      = os.getenv("UPSTAGE_MODEL", "solar-pro2")
LOCAL_MODEL_ID     = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
# UI 셀렉트박스 선택지. "Qwen (로컬)"은 로컬 백엔드로, "solar-*"는 Upstage API로 라우팅된다.
# select_model()이 이 라벨을 (backend, upstage_model)로 변환한다.
MODEL_CHOICES      = ["Qwen (로컬)", "solar-pro2", "solar-pro3", "solar-pro"]
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

답변 구조 (해당 항목만 포함하세요):
1. **접근 전략** — 어떤 방법을 왜 선택해야 하는지 (모델·알고리즘·피처 관점)
2. **예상 함정** — 초보자가 흔히 빠지는 실수나 간과하는 제약
3. **참고 코드/수식** — 코드블록이나 수식이 도움될 때만 포함

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
    "답변 구조 (해당 항목만 포함하세요):\n"
    "1. 접근 전략 — 어떤 방법을 왜 선택해야 하는지 (모델·알고리즘·피처 관점)\n"
    "2. 예상 함정 — 초보자가 흔히 빠지는 실수나 간과하는 제약\n"
    "3. 참고 코드/수식 — 코드블록이나 수식이 도움될 때만 포함\n\n"
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

def _strip_object_noise(text: str) -> str:
    """[object Object]/undefined 노이즈를 제거한다.

    이 토큰만으로 이뤄진 줄(공백·구두점·들여쓰기·리스트마커·백틱 펜스만 남는 줄)은 통째로 삭제한다.
    코드블록·리스트·들여쓰기 등 어디에 박혀도 '의미 없는 줄'이라 형태와 무관하게 줄 단위로 잡는 게
    가장 견고하다(형태별 정규식을 쫓다 새 형태가 계속 빠져나가던 것을 한 규칙으로 통합).
    다른 내용과 같은 줄에 섞인 경우엔 토큰만 지우고 줄은 보존한다."""
    # JS 객체 직렬화 산물 변종. 모델(특히 solar-pro/pro2)이 컨텍스트 노이즈를 따라 뱉는 형태들.
    noise = r"(?:\[object Object\]|\[object [A-Za-z]+\]|\bundefined\b|\bNaN\b|\bnull\b)"
    # 1) 노이즈 토큰 + 그 주변 구두점만으로 이뤄진 줄 → 줄 전체 삭제(줄바꿈 포함).
    only_noise_line = rf"(?m)^[ \t]*(?:{noise}[\s,;:.\-•·`]*)+$\n?"
    text = re.sub(only_noise_line, "", text)
    # 2) 다른 내용과 섞인 줄에 남은 토큰은 토큰만 제거.
    text = re.sub(rf"{noise}[,\s]*", "", text)
    return text


def _clean_text(text: str) -> str:
    """문서 청크의 노이즈 문자를 정규화한다."""
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = _strip_object_noise(text)  # JS 객체 문자열화 산물 제거 (입력 차단)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\S) {2,}", " ", text)  # 비공백 뒤의 중복 공백만 압축 (줄머리 들여쓰기 보존 → 중첩 목록 유지)
    return text.strip()


def _clean_answer(text: str) -> str:
    """LLM 출력의 잔여 노이즈를 제거한다."""
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    # 빈 코드블록을 펜스 쌍이 온전할 때 먼저 제거한다. _has_real_content가 노이즈를 무시하므로
    # 본문이 [object Object]뿐인 블록은 여는·닫는 펜스째 통째로 사라진다(비대칭 펜스 방지).
    text = _strip_empty_codeblocks(text)
    # 남은 노이즈(펜스 밖, 리스트·들여쓰기에 박힌 것)를 줄 단위로 제거. _clean_text와 동일 규칙.
    text = _strip_object_noise(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\S) {2,}", " ", text)  # 비공백 뒤의 중복 공백만 압축 (줄머리 들여쓰기 보존 → 중첩 목록 유지)
    # 제목·단락 직후 목록 항목 앞 빈 줄 보장 — CommonMark/Streamlit 렌더러는 빈 줄 없이
    # 바로 이어지는 목록을 bullet list로 파싱하지 못해 `- 항목`이 일반 텍스트로 출력됨.
    text = re.sub(r"(?m)(?<=[^\n])\n([ \t]*[-*+] )", r"\n\n\1", text)
    text = re.sub(r"(?m)(?<=[^\n])\n([ \t]*\d+\. )", r"\n\n\1", text)
    text = _remove_trailing_empty_headings(text)
    return text.strip()


def _has_real_content(body: str) -> bool:
    """코드블록 본문에 의미 있는 내용(영숫자·한글 등 단어 문자)이 있는지.
    공백·구두점만이거나, 노이즈([object Object]/undefined)만이면 False — 그 노이즈의 'object'를
    실제 내용으로 오인하면 빈 블록이 안 지워진다. 노이즈를 먼저 비운 뒤 판정한다."""
    body = _strip_object_noise(body)
    return bool(re.search(r"[0-9A-Za-z가-힣]", body))


def _strip_empty_codeblocks(text: str) -> str:
    """본문이 사실상 비어버린 코드블록(빈 박스)을 제거한다.

    노이즈([object Object]/undefined)를 지운 뒤 코드블록 본문이 공백·구두점만 남는 경우가 있다.
    펜스(```)를 순서대로 짝지어, 열고 닫힌 블록의 본문이 비면 통째로 삭제한다. 닫히지 않은 채
    끝난 펜스(Solar가 코드블록을 안 닫고 종료)는, 본문이 비면 제거하고 내용이 있으면 닫아준다.
    정규식 한 방으로 끝줄 펜스를 지우면 정상 코드의 '닫는 펜스'까지 오삭제되므로 토큰 단위로 처리한다."""
    # 펜스줄(```...)을 기준으로 쪼갠다. 토큰을 순회하며 펜스를 만날 때마다 in/out을 토글해
    # '여는 펜스 → 본문 → 닫는 펜스'를 짝짓는다. 본문이 비면 그 블록(여는·닫는 펜스 포함)을 버린다.
    # 선행 공백·리스트 마커를 허용 — 들여쓰기된 펜스(  ```)와 Solar가 만든 비표준 '리스트마커+펜스'
    # 한 줄(  - ```)도 잡아야 빈 박스가 안 남는다. 빈 블록이면 마커째 제거(그 항목 자체가 노이즈).
    tokens = re.split(r"(?m)^([ \t]*(?:[-*+]|\d+\.)?[ \t]*```[^\n]*)$", text)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        is_fence = i % 2 == 1  # split의 캡처(펜스줄)는 항상 홀수 인덱스
        if not is_fence:
            out.append(tok)
            i += 1
            continue
        # 여는 펜스. 본문(i+1)과 닫는 펜스(i+2) 존재 여부 확인.
        body = tokens[i + 1] if i + 1 < len(tokens) else ""
        has_close = i + 2 < len(tokens)
        if not _has_real_content(body):
            # 빈 블록: 여는 펜스·빈 본문·(있으면)닫는 펜스를 통째로 버린다.
            i += 3 if has_close else 2
        elif has_close:
            out.append(tok); out.append(body); out.append(tokens[i + 2])
            i += 3
        else:
            # 닫히지 않은 블록에 내용이 있음 → Solar가 닫는 펜스를 빠뜨린 것.
            # 펜스를 닫아 코드블록을 완성한다(빈 박스가 뒤 텍스트를 삼키는 것 방지).
            # ponytail: 코드/일반텍스트 구분은 휴리스틱이라 안 함 — 빈 칸만 확실히 제거.
            out.append(tok); out.append(body.rstrip() + "\n```")
            i += 2
    return "".join(out)


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
        chunk_size=1000,
        chunk_overlap=150,
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
                "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
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
    if OFFLINE_MODE:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
        )
    from langchain_upstage import UpstageEmbeddings
    return UpstageEmbeddings(
        api_key=UPSTAGE_API_KEY,
        model="solar-embedding-1-large",
    )


def _resolve_choice(choice: str | None) -> tuple[str, str]:
    """UI 셀렉트박스 라벨을 (backend, upstage_model)로 변환한다.
    choice가 None이면 전역 설정(LLM_BACKEND/UPSTAGE_MODEL)을 그대로 쓴다.
    "solar-*" 라벨이면 upstage 백엔드 + 그 별칭, 그 외("Qwen (로컬)" 등)면 전역 로컬 백엔드.
    """
    if not choice:
        return LLM_BACKEND, UPSTAGE_MODEL
    if choice.startswith("solar"):
        return "upstage", choice
    # 로컬 선택 — 전역 LLM_BACKEND가 로컬(qwen/vllm)이면 그걸 쓰고, 아니면 qwen 기본.
    backend = LLM_BACKEND if LLM_BACKEND in _LOCAL_BACKENDS else "qwen"
    return backend, UPSTAGE_MODEL


def _get_upstage_llm(model: str | None = None):
    from langchain_upstage import ChatUpstage
    return ChatUpstage(api_key=UPSTAGE_API_KEY, model=model or UPSTAGE_MODEL)


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


def warm_up_llm(model: str | None = None, progress=None) -> None:
    """로컬 LLM(LLaMA/Qwen)을 미리 로딩해 캐시에 채운다.

    첫 질문이 '모델 로딩 + 추론'을 한꺼번에 떠안아 WebSocket 타임아웃이 나는 것을 막는다.
    문서 인덱싱 직후(이미 spinner가 도는 구간)에 호출하면 실제 질문은 추론만 한다.
    Upstage(API)·OFFLINE이면 LLM 로딩이 없으나, 리랭커는 공통으로 미리 캐시한다.
    model: UI 선택 라벨. 로컬 백엔드로 풀릴 때만 LLM을 데운다(Solar 선택 시 무의미).
    progress: 선택적 콜백 (label: str) -> None. 단계별 진행 상황을 UI에 표시한다.
    """
    def _step(label: str) -> None:
        if progress is not None:
            progress(label)

    # 리랭커도 첫 질문에서 분리(질문마다 재로딩하던 것을 미리 캐시).
    _step("리랭커 모델 로딩 중 (cross-encoder)...")
    _rerank_with_cross_encoder("warm up", [Document(page_content="warm up")], top_k=1)
    backend, _ = _resolve_choice(model)
    if not OFFLINE_MODE and backend in _LOCAL_BACKENDS:
        _step(f"LLM 모델 로딩 중 ({backend})...")
        _get_llama_llm()  # 캐시(_llama_llm_cache)를 채운다


def answer_question(vectorstore: FAISS, question: str, progress=None,
                    model: str | None = None) -> dict[str, Any]:
    """{'answer': str, 'sources': list[Document]} 반환.

    progress: 선택적 콜백 (label: str) -> None. 각 단계 시작 시 호출돼 진행 상황을
    UI에 표시한다. None이면(기본) 아무 것도 하지 않아 기존 호출부와 호환된다.
    model: UI 셀렉트박스 라벨("Qwen (로컬)" | "solar-pro2" 등). None이면 전역 설정 사용.
    """
    def _step(label: str) -> None:
        if progress is not None:
            progress(label)

    backend, upstage_model = _resolve_choice(model)

    # ── 캐시 확인 ────────────────────────────────────────────────────────────
    # 모델 선택까지 키에 포함 — 같은 질문이라도 모델이 다르면 별도 답변이어야 한다.
    cache_key = (id(vectorstore), question, backend, upstage_model)
    if cache_key in _answer_cache:
        _step("캐시된 답변 사용")
        return _answer_cache[cache_key]

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    is_local_backend = backend in _LOCAL_BACKENDS
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
        _step(f"답변 생성 ({backend})")
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
        _step(f"답변 생성 ({upstage_model})")
        raw_answer = _run_chain(_get_upstage_llm(upstage_model), _get_upstage_prompt())
    except Exception as exc:
        import warnings
        warnings.warn(f"[recsys-coach] Upstage 호출 실패 ({exc!r}), Llama-3으로 폴백합니다.")
        _step("Solar 실패 — Llama-3 폴백 생성")
        raw_answer = _run_chain(_get_llama_llm(), _get_llama_prompt())

    result = {"answer": _clean_answer(raw_answer), "sources": context_docs}
    _answer_cache[cache_key] = result
    return result
