from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


class Checklist:
    """분석 진행 상황을 □/✓ 체크리스트로 실시간 표시한다.

    start(label)로 항목을 □ 상태로 추가하고, complete()로 직전 항목을 ✓로 바꾼다.
    항목이 바뀔 때마다 placeholder 전체를 다시 렌더링하므로 화면이 자동 갱신된다.
    """

    def __init__(self, placeholder: st.delta_generator.DeltaGenerator) -> None:
        self._placeholder = placeholder
        self._items: list[tuple[str, bool]] = []  # (label, done)

    def start(self, label: str) -> None:
        self._items.append((label, False))
        self._render()

    def complete(self) -> None:
        for i in range(len(self._items) - 1, -1, -1):
            label, done = self._items[i]
            if not done:
                self._items[i] = (label, True)
                break
        self._render()

    def _render(self) -> None:
        lines = [f"{'✓' if done else '□'} {label}" for label, done in self._items]
        self._placeholder.markdown("\n".join(f"- {ln}" for ln in lines))


# ── 중앙 정렬 표 렌더링 ────────────────────────────────────────────────
_CENTERED_CELL_PROPS = [
    ("text-align", "center !important"),
    ("vertical-align", "middle !important"),
    ("width", "auto !important"),
    ("white-space", "nowrap !important"),
]
_CENTERED_TABLE_STYLES = [
    {"selector": "", "props": [("width", "auto !important"), ("table-layout", "auto !important")]},
    {"selector": "table", "props": [("width", "auto !important"), ("table-layout", "auto !important")]},
    {"selector": "th", "props": _CENTERED_CELL_PROPS},
    {"selector": "td", "props": _CENTERED_CELL_PROPS},
    {"selector": "thead th", "props": _CENTERED_CELL_PROPS},
    {"selector": "th.col_heading", "props": _CENTERED_CELL_PROPS},
    {"selector": "th.blank", "props": _CENTERED_CELL_PROPS},
]


def _inject_centered_table_css() -> None:
    """표 텍스트 정렬을 중앙으로 강제한다."""
    st.markdown(
        """
        <style>
        [data-testid="stTable"] {
            width: fit-content !important;
            max-width: 100% !important;
            overflow-x: auto !important;
        }
        [data-testid="stTable"] table, table {
            width: auto !important;
            table-layout: auto !important;
        }
        th, td,
        [data-testid="stTable"] th, [data-testid="stTable"] td {
            text-align: center !important;
            vertical-align: middle !important;
            white-space: nowrap !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_centered_table(table: pd.DataFrame) -> None:
    """표 전체를 중앙 정렬해 st.table로 렌더링한다 (인덱스 숨김)."""
    styler = (
        table.style
        .set_properties(**{"text-align": "center !important", "vertical-align": "middle !important"})
        .set_table_styles(_CENTERED_TABLE_STYLES, overwrite=False)
        .hide(axis="index")
    )
    st.table(styler)


def detect_compute_backend() -> str:
    """'gpu' 또는 'cpu' 반환."""
    try:
        import cudf  # noqa: F401
        import torch
        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        pass
    return "cpu"


# ── 컬럼 해석 & 포맷 헬퍼 ──────────────────────────────────────────────
_ITEM_CANDIDATES = ["product_id", "item_id", "product", "sku", "asin", "itemid"]
_USER_CANDIDATES = ["user_id", "user", "customer_id", "visitorid", "uid", "userid"]
_EVENT_CANDIDATES = ["event_type", "event", "action", "behavior", "event_name"]


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _fmt_int(n) -> str:
    return f"{int(n):,}"


def _fmt_pct(x, digits: int = 2) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def _fmt_float(x, digits: int = 2) -> str:
    return "—" if pd.isna(x) else f"{x:,.{digits}f}"


def _gini(values) -> float:
    """인기 분포의 지니계수 (0=완전 균등, 1=완전 쏠림)."""
    arr = np.sort(np.asarray(values, dtype="float64"))
    n = arr.size
    if n == 0 or arr.sum() == 0:
        return float("nan")
    cum = np.cumsum(arr)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n


# ── 추천 특화 분석 표 ─────────────────────────────────────────────────
def _funnel_table(df: pd.DataFrame, event_col: str):
    """view → cart → purchase 전환 퍼널 표."""
    order = ["view", "cart", "purchase"]
    label_map = {"view": "조회 (view)", "cart": "장바구니 (cart)", "purchase": "구매 (purchase)"}
    vc = df[event_col].astype(str).str.lower().value_counts()
    stages = [s for s in order if s in vc.index]
    if len(stages) < 2:
        return None
    first = int(vc[stages[0]])
    rows = []
    prev = None
    for s in stages:
        cnt = int(vc[s])
        step = "—" if prev is None else _fmt_pct(cnt / prev if prev else float("nan"))
        overall = _fmt_pct(cnt / first if first else float("nan"))
        rows.append({
            "단계": label_map.get(s, s),
            "이벤트 수": _fmt_int(cnt),
            "직전 단계 대비": step,
            "조회 대비": overall,
        })
        prev = cnt
    return pd.DataFrame(rows)


def _longtail_table(df: pd.DataFrame, item_col: str):
    """아이템 인기 분포 (롱테일·인기 편향) 표."""
    pop = df[item_col].value_counts()
    n_items = int(pop.size)
    total = int(pop.sum())
    if n_items == 0 or total == 0:
        return None
    counts = pop.sort_values(ascending=False).to_numpy()
    cum = counts.cumsum()

    def top_share(frac: float) -> float:
        k = max(1, int(round(n_items * frac)))
        return cum[k - 1] / total

    once = int((pop == 1).sum())
    rows = [
        {"지표": "전체 아이템 수", "값": _fmt_int(n_items)},
        {"지표": "총 상호작용 수", "값": _fmt_int(total)},
        {"지표": "상위 1% 아이템 점유율", "값": _fmt_pct(top_share(0.01))},
        {"지표": "상위 5% 아이템 점유율", "값": _fmt_pct(top_share(0.05))},
        {"지표": "상위 10% 아이템 점유율", "값": _fmt_pct(top_share(0.10))},
        {"지표": "상위 20% 아이템 점유율", "값": _fmt_pct(top_share(0.20))},
        {"지표": "지니계수 (인기 편향)", "값": _fmt_float(_gini(counts), 3)},
        {"지표": "1회만 등장한 아이템 비율", "값": _fmt_pct(once / n_items)},
    ]
    return pd.DataFrame(rows)


def _sparsity_table(df: pd.DataFrame, user_col: str, item_col: str):
    """사용자×아이템 상호작용 희소성·콜드스타트 표."""
    n_users = int(df[user_col].nunique())
    n_items = int(df[item_col].nunique())
    if n_users == 0 or n_items == 0:
        return None
    n_inter = int(len(df))
    pairs = int(df[[user_col, item_col]].drop_duplicates().shape[0])
    density = pairs / (n_users * n_items)
    per_user = df.groupby(user_col).size()
    per_item = df.groupby(item_col).size()
    users_once = int((per_user == 1).sum())
    items_once = int((per_item == 1).sum())
    rows = [
        {"지표": "사용자 수", "값": _fmt_int(n_users)},
        {"지표": "아이템 수", "값": _fmt_int(n_items)},
        {"지표": "총 상호작용 수", "값": _fmt_int(n_inter)},
        {"지표": "고유 (사용자·아이템) 쌍", "값": _fmt_int(pairs)},
        {"지표": "행렬 밀도 (density)", "값": f"{density * 100:.4f}%"},
        {"지표": "희소도 (sparsity)", "값": f"{(1 - density) * 100:.4f}%"},
        {"지표": "사용자당 평균 상호작용", "값": _fmt_float(per_user.mean())},
        {"지표": "사용자당 중앙값 상호작용", "값": _fmt_float(per_user.median())},
        {"지표": "아이템당 평균 상호작용", "값": _fmt_float(per_item.mean())},
        {"지표": "1회 상호작용 사용자 비율 (콜드스타트)", "값": _fmt_pct(users_once / n_users)},
        {"지표": "1회 상호작용 아이템 비율 (콜드스타트)", "값": _fmt_pct(items_once / n_items)},
    ]
    return pd.DataFrame(rows)


def _eda_recsys(df: pd.DataFrame, checklist: Checklist) -> None:
    """추천 시스템 특화 분석만 표로 출력한다 (전환 퍼널·롱테일·희소성)."""
    if not isinstance(df, pd.DataFrame):
        try:
            df = df.to_pandas()  # cudf -> pandas
        except Exception:
            df = pd.DataFrame(df)

    # 어떤 파일이든 항상 출력하는 기본 개요
    checklist.start("데이터 개요")
    st.subheader("데이터 개요")
    overview = pd.DataFrame({
        "타입": df.dtypes.astype(str),
        "고유값": df.nunique(),
        "결측 수": df.isnull().sum(),
        "결측률(%)": (df.isnull().sum() / len(df) * 100).round(2),
    })
    _render_centered_table(overview.reset_index(names="컬럼"))
    checklist.complete()

    event_col = _first_present(df, _EVENT_CANDIDATES)
    item_col = _first_present(df, _ITEM_CANDIDATES)
    user_col = _first_present(df, _USER_CANDIDATES)

    rendered = False

    checklist.start("전환 퍼널 (view → cart → purchase)")
    try:
        funnel = _funnel_table(df, event_col) if event_col else None
        if funnel is not None:
            st.subheader("전환 퍼널")
            st.caption("행동 단계별 이벤트 수와 단계 간 전환율 — 추천이 실제 구매로 이어지는 구간을 진단합니다.")
            _render_centered_table(funnel)
            rendered = True
    except Exception as e:
        st.warning(f"전환 퍼널 분석 실패: {e}")
    checklist.complete()

    checklist.start("롱테일·인기 편향")
    try:
        longtail = _longtail_table(df, item_col) if item_col else None
        if longtail is not None:
            st.subheader("롱테일·인기 편향")
            st.caption("소수 인기 아이템에 상호작용이 쏠리는 정도 — 인기 편향 보정·롱테일 추천 전략의 근거.")
            _render_centered_table(longtail)
            rendered = True
    except Exception as e:
        st.warning(f"롱테일 분석 실패: {e}")
    checklist.complete()

    checklist.start("상호작용 희소성·콜드스타트")
    try:
        sparsity = _sparsity_table(df, user_col, item_col) if (user_col and item_col) else None
        if sparsity is not None:
            st.subheader("상호작용 희소성")
            st.caption("사용자×아이템 행렬의 밀도와 콜드스타트 비중 — 협업 필터링 난이도와 콜드스타트 대응 필요성.")
            _render_centered_table(sparsity)
            rendered = True
    except Exception as e:
        st.warning(f"희소성 분석 실패: {e}")
    checklist.complete()

    if not rendered:
        st.info("추천 특화 분석에 필요한 컬럼(event_type · item_id · user_id 등)을 찾지 못했습니다.\n"
                f"감지된 컬럼 — event: {event_col}, item: {item_col}, user: {user_col}")


def generate_eda(df: pd.DataFrame, checklist: Checklist) -> None:
    """추천 특화 EDA 진입점."""
    _inject_centered_table_css()
    backend = detect_compute_backend()

    with st.sidebar:
        label = "GPU: RTX 3090" if backend == "gpu" else "CPU"
        st.info(f"연산 백엔드: **{label}**")

    _eda_recsys(df, checklist)
