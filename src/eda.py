import time

import pandas as pd
import streamlit as st


class Checklist:
    """분석 진행 상황을 체크리스트로 실시간 표시한다(단계별 소요 시간 포함).

    start(label)로 항목을 추가하면 진행 중(⏳) 상태로 나타나고,
    complete()를 호출하면 직전 항목이 완료(✓)로 바뀌며 소요 시간이 붙는다.
    항목이 추가/완료될 때마다 placeholder 전체를 다시 렌더링해 화면이 자동 갱신된다.
    호출부 시그니처(start/complete)는 그대로라 모든 사용처에 자동 적용된다.
    """

    def __init__(self, placeholder: st.delta_generator.DeltaGenerator) -> None:
        self._placeholder = placeholder
        self._items: list[list] = []  # [label, done, start_ts, elapsed]
        self._t0 = time.perf_counter()  # 전체 시작 시각(누적 경과 표시용)

    def start(self, label: str) -> None:
        """새 분석 항목을 진행 중 상태로 추가하고 화면을 갱신한다."""
        self._items.append([label, False, time.perf_counter(), None])
        self._render()

    def complete(self) -> None:
        """가장 최근에 시작한 항목을 완료로 바꾸고 소요 시간을 기록·갱신한다."""
        for item in reversed(self._items):
            if not item[1]:
                item[1] = True
                item[3] = time.perf_counter() - item[2]
                break
        self._render()

    def _render(self) -> None:
        lines = []
        for label, done, _start, elapsed in self._items:
            if done:
                lines.append(f"✓ {label} — `{elapsed:.1f}s`")
            else:
                lines.append(f"⏳ {label} …")
        total = time.perf_counter() - self._t0
        body = "\n".join(f"- {ln}" for ln in lines)
        self._placeholder.markdown(f"{body}\n\n**총 경과: `{total:.1f}s`**")


# 범주형 ID로 볼 고유값 상한 — 이보다 적으면 분포 막대그래프가 의미 있음
_CATEGORICAL_ID_MAX_UNIQUE = 50


def _is_id_like(col: str) -> bool:
    """ID성(식별자) 컬럼인지 컬럼명으로 판별한다.

    숫자로 저장돼 있어도 product_id/user_id/category_id 처럼 본질이 식별자인
    컬럼은 연속형 히스토그램이 의미가 없으므로 별도로 분류한다.
    판별은 컬럼명만으로 한다 — price 같은 연속 수치가 고유값이 많다는 이유로
    식별자로 오분류되던 문제(고유값 비율 휴리스틱)를 제거했다.
    """
    name = col.lower()
    return name.endswith("_id") or name == "id" or "session" in name


def _analysis_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """(연속형, 범주형 ID, 식별자 ID) 세 그룹으로 수치 컬럼을 분류한다.

    - 연속형: 히스토그램 대상 (price 등)
    - 범주형 ID: 고유값이 적어 분포 막대그래프가 유의미 (category_id 등)
    - 식별자 ID: 고유값이 많아 분포가 무의미, 고유값 개수만 요약 (product_id, user_id 등)
    """
    numeric = df.select_dtypes("number").columns.tolist()
    continuous = [c for c in numeric if not _is_id_like(c)]
    id_cols = [c for c in numeric if _is_id_like(c)]
    categorical_id = [c for c in id_cols if df[c].nunique(dropna=True) <= _CATEGORICAL_ID_MAX_UNIQUE]
    identifier_id = [c for c in id_cols if df[c].nunique(dropna=True) > _CATEGORICAL_ID_MAX_UNIQUE]
    return continuous, categorical_id, identifier_id


def _partition_object_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """object 컬럼을 (시계열 후보, 세션, 일반) 세 그룹으로 분류한다.

    - 시계열 후보(datetime_cols): 컬럼명에 "time" 또는 "date" 포함 (event_time 등).
      문자열로 들어온 시각 컬럼이 대상이며, 실제 파싱 성공 여부는 호출부에서
      to_datetime(errors="coerce")로 확인한다(여기서는 이름만 본다).
    - 세션(session_cols): 컬럼명에 "session" 포함 (user_session 등).
    - 일반(plain_cols): 나머지 (event_type / category_code / brand 등) — 상위 N value_counts가 유의미.

    이름이 "session"과 "time"/"date"에 모두 걸리면 session 우선. 버킷은 상호배타라
    한 컬럼이 두 번 렌더링되지 않는다. 컬럼 순서는 보존한다.
    """
    obj_cols = df.select_dtypes("object").columns.tolist()
    datetime_cols: list[str] = []
    session_cols: list[str] = []
    plain_cols: list[str] = []
    for col in obj_cols:
        name = col.lower()
        if "session" in name:  # session 우선
            session_cols.append(col)
        elif "time" in name or "date" in name:
            datetime_cols.append(col)
        else:
            plain_cols.append(col)
    return datetime_cols, session_cols, plain_cols


# 값별 개수 표에 기본 노출할 상위 행 수 (연속형은 고유값이 많아 전체 나열 시 표가 폭주)
_VALUE_TABLE_TOP_N = 20
# user_session 세션당 이벤트 수, price 값별 개수에 적용할 상위 행 수
_SESSION_PRICE_TOP_N = 10
_TOP10_DISTRIBUTION_COLUMNS = ("event_type", "category_code", "brand", "event_time")
_TOP10_DISTRIBUTION_N = 10
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
    {"selector": "tbody th", "props": _CENTERED_CELL_PROPS},
    {"selector": "th.col_heading", "props": _CENTERED_CELL_PROPS},
    {"selector": "th.row_heading", "props": _CENTERED_CELL_PROPS},
    {"selector": "th.blank", "props": _CENTERED_CELL_PROPS},
]


def _inject_centered_table_css() -> None:
    """Streamlit/HTML/AgGrid 표 텍스트 정렬만 중앙으로 강제한다."""
    st.markdown(
        """
        <style>
        [data-testid="stTable"] {
            width: fit-content !important;
            max-width: 100% !important;
            overflow-x: auto !important;
        }

        [data-testid="stTable"] table,
        table {
            width: auto !important;
            table-layout: auto !important;
        }

        th,
        td,
        [data-testid="stTable"] th,
        [data-testid="stTable"] td,
        [data-testid="stDataFrame"] th,
        [data-testid="stDataFrame"] td,
        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] [role="rowheader"],
        [data-testid="stDataFrame"] [role="gridcell"],
        .ag-header-cell,
        .ag-header-cell-label,
        .ag-header-cell-text,
        .ag-cell,
        .ag-cell-value {
            text-align: center !important;
            vertical-align: middle !important;
            justify-content: center !important;
            align-items: center !important;
            width: auto !important;
            white-space: nowrap !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _centered_table_styler(data, hide_index: bool = False):
    """Pandas Styler로 헤더, 데이터 셀, 인덱스를 모두 중앙 정렬한다."""
    table = data.to_frame() if isinstance(data, pd.Series) else data
    id_formatters = {
        col: _format_id_value_for_display for col in table.columns if _is_id_like(str(col))
    }
    styler = table.style
    if id_formatters:
        styler = styler.format(id_formatters)
    if _is_id_like(str(table.index.name or "")):
        styler = styler.format_index(_format_id_value_for_display, axis=0)

    styler = styler.set_properties(
        **{
            "text-align": "center !important",
            "vertical-align": "middle !important",
        }
    ).set_table_styles(_CENTERED_TABLE_STYLES, overwrite=False)
    if hide_index:
        styler = styler.hide(axis="index")
    return styler


def _render_centered_table(table: pd.DataFrame) -> None:
    """표 전체를 중앙 정렬해 st.table로 렌더링한다.

    Pandas Styler로 th/td/index 정렬을 모두 중앙으로 맞춘다. 인덱스는 숨긴다.
    """
    st.table(_centered_table_styler(table, hide_index=True))


def _render_centered_dataframe(data) -> None:
    """DataFrame/Series 표도 헤더, 인덱스, 데이터 셀을 모두 중앙 정렬한다."""
    st.table(_centered_table_styler(data))


def _format_id_value_for_display(value):
    """ID처럼 쓰이는 정수형 float 값을 표시용 문자열로 바꾼다."""
    if pd.isna(value):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer():
        return str(int(number))
    return str(value)


def _value_counts_table(counts: pd.Series, value_label: str, top_n: int = _VALUE_TABLE_TOP_N) -> pd.DataFrame:
    """value_counts 결과(Series)를 (값, 개수) 2열 표로 변환한다.

    counts는 {값: 개수} 형태의 pandas Series(이미 .value_counts() 수행된 것)를 받는다.
    개수 내림차순 상위 top_n개만 담는다 — 연속형처럼 고유값이 많을 때 표 폭주를 막는다.
    """
    top = counts.sort_values(ascending=False).head(top_n)
    values = top.index
    if _is_id_like(value_label):
        values = [_format_id_value_for_display(value) for value in values]
    return pd.DataFrame({value_label: values, "개수": top.values})


def _top10_distribution_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """요청된 주요 컬럼의 상위 10 값 분포를 표로 만든다."""
    if col == "event_time":
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().any():
            counts = parsed.dt.floor("D").value_counts(dropna=False)
            return _value_counts_table(counts, f"{col} (일자)", top_n=_TOP10_DISTRIBUTION_N)

    counts = df[col].value_counts(dropna=False)
    return _value_counts_table(counts, col, top_n=_TOP10_DISTRIBUTION_N)


def _render_top10_distribution_tables(df: pd.DataFrame, checklist: Checklist) -> set[str]:
    """event_type/category_code/brand/event_time 상위 10 값 분포 표를 출력한다."""
    rendered_cols: set[str] = set()
    for col in _TOP10_DISTRIBUTION_COLUMNS:
        if col not in df.columns:
            continue
        checklist.start(f"`{col}` 값 분포 표")
        st.subheader(f"`{col}` 값 분포 (상위 {_TOP10_DISTRIBUTION_N})")
        _render_centered_table(_top10_distribution_table(df, col))
        checklist.complete()
        rendered_cols.add(col)
    return rendered_cols


def detect_compute_backend() -> str:
    """'gpu' 또는 'cpu' 반환"""
    try:
        import cudf  # noqa: F401
        import torch
        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        pass
    return "cpu"


def _eda_gpu(df: pd.DataFrame, checklist: Checklist) -> None:
    import cudf
    import plotly.express as px

    checklist.start("GPU 데이터 로드")
    gdf = cudf.DataFrame.from_pandas(df)
    checklist.complete()

    continuous_cols, categorical_id_cols, identifier_id_cols = _analysis_columns(df)

    checklist.start("기초 통계")
    st.subheader("기초 통계")
    if continuous_cols:
        _render_centered_dataframe(gdf[continuous_cols].describe().to_pandas().T)
    else:
        st.info("기초 통계 대상 연속형 수치 컬럼이 없습니다.")
    checklist.complete()

    num_cols = df.select_dtypes("number").columns.tolist()
    if len(num_cols) >= 2:
        checklist.start("상관관계 히트맵")
        corr = gdf[num_cols].corr().to_pandas()
        st.subheader("상관관계 히트맵")
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
        checklist.complete()

    checklist.start("결측값 분석")
    st.subheader("결측값")
    missing = gdf.isnull().sum().to_pandas().rename("결측 수")
    _render_centered_dataframe(missing[missing > 0])
    checklist.complete()

    datetime_cols, session_cols, plain_cols = _partition_object_columns(df)
    rendered_top10_cols = _render_top10_distribution_tables(df, checklist)
    plain_cols = [col for col in plain_cols if col not in rendered_top10_cols]

    # 시계열 후보(event_time 등): datetime 파싱 성공 시 일별 이벤트 수 라인 차트,
    # 실패 시 plain_cols로 되돌려 일반 막대그래프 처리.
    # cuDF의 to_datetime은 Series 입력에 errors="coerce"를 지원하지 않고 파싱 실패 시
    # 예외를 던지므로, try/except로 감싸 성공 여부로 폴백을 결정한다(CPU 경로의 coerce와 동일 효과).
    for col in datetime_cols:
        try:
            parsed = cudf.to_datetime(gdf[col])
        except Exception:
            if col not in rendered_top10_cols:
                plain_cols.append(col)  # 파싱 불가 → 기존 막대 처리로 폴백
            continue
        checklist.start(f"`{col}` 시계열 (일별 이벤트 수)")
        st.subheader(f"`{col}` 시계열 (일별 이벤트 수)")
        daily = parsed.dropna().dt.floor("D").value_counts().to_pandas().sort_index()
        fig = px.line(x=daily.index, y=daily.values, labels={"x": col, "y": "event count"})
        st.plotly_chart(fig, use_container_width=True)
        checklist.complete()

    # 세션 컬럼(user_session 등): 세션 해시 상위 N은 무의미 → 세션당 이벤트 수 분포(값별 개수 표).
    # 집계는 GPU(groupby)에서, 결과(세션 개수 단위)만 host로 내려 표 작성.
    for col in session_cols:
        checklist.start(f"`{col}` 세션당 이벤트 수")
        st.subheader(f"`{col}` 세션당 이벤트 수 (상위 {_SESSION_PRICE_TOP_N})")
        events_per_session = gdf.groupby(col).size().to_pandas()  # NaN 세션 키는 자동 제외
        if events_per_session.empty:
            st.info("세션 데이터가 없습니다.")
            checklist.complete()
            continue
        table = _value_counts_table(events_per_session.value_counts(), "세션당 이벤트 수", top_n=_SESSION_PRICE_TOP_N)
        _render_centered_table(table)
        checklist.complete()

    # 일반 object 컬럼(event_type / category_code / brand 등): 상위 10개 값 분포 막대
    for col in plain_cols[:5]:
        checklist.start(f"`{col}` 값 분포")
        st.subheader(f"`{col}` 값 분포 (상위 10)")
        counts = gdf[col].value_counts().head(10).to_pandas()
        _render_centered_table(_value_counts_table(counts, col))
        checklist.complete()

    # 연속형 수치 컬럼: 값별 개수 표 (price 등). 고유값이 많아 상위 N개만 표시.
    for col in continuous_cols:
        checklist.start(f"`{col}` 값별 개수")
        st.subheader(f"`{col}` 값별 개수 (상위 {_SESSION_PRICE_TOP_N})")
        counts = gdf[col].value_counts().head(_SESSION_PRICE_TOP_N).to_pandas()
        table = _value_counts_table(counts, col)
        _render_centered_table(table)
        checklist.complete()

    # 범주형 ID 컬럼: 고유값이 적어 분포 막대그래프가 유의미 (category_id 등)
    for col in categorical_id_cols:
        checklist.start(f"`{col}` 값 분포")
        st.subheader(f"`{col}` 값 분포 (상위 10)")
        counts = gdf[col].value_counts().head(10).to_pandas()
        labels = [_format_id_value_for_display(value) for value in counts.index]
        fig = px.bar(x=labels, y=counts.values, labels={"x": col, "y": "count"})
        st.plotly_chart(fig, use_container_width=True)
        checklist.complete()

    # 식별자 ID 컬럼: 고유값이 많아 분포가 무의미, 고유값 개수만 요약 (product_id, user_id 등)
    if identifier_id_cols:
        checklist.start("ID 컬럼 요약 (고유값 개수)")
        summary = pd.DataFrame(
            {
                "고유값 개수": [int(gdf[c].nunique()) for c in identifier_id_cols],
                "결측 수": [int(gdf[c].isnull().sum()) for c in identifier_id_cols],
            },
            index=identifier_id_cols,
        )
        st.subheader("ID 컬럼 요약")
        st.caption("product_id·user_id 등 식별자 컬럼은 히스토그램 대신 고유값 개수로 요약합니다.")
        _render_centered_dataframe(summary)
        checklist.complete()


def _eda_cpu(df: pd.DataFrame, checklist: Checklist) -> None:
    import plotly.express as px

    continuous_cols, categorical_id_cols, identifier_id_cols = _analysis_columns(df)

    checklist.start("기초 통계")
    st.subheader("기초 통계")
    if continuous_cols:
        _render_centered_dataframe(df[continuous_cols].describe().T)
    else:
        st.info("기초 통계 대상 연속형 수치 컬럼이 없습니다.")
    checklist.complete()

    checklist.start("결측값 분석")
    missing = df.isnull().sum().rename("결측 수")
    missing = missing[missing > 0]
    st.subheader("결측값")
    if missing.empty:
        st.success("결측값 없음")
    else:
        _render_centered_dataframe(missing)
    checklist.complete()

    num_cols_all = df.select_dtypes("number").columns.tolist()
    if len(num_cols_all) >= 2:
        checklist.start("상관관계 히트맵")
        corr = df[num_cols_all].corr()
        st.subheader("상관관계 히트맵")
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
        checklist.complete()

    datetime_cols, session_cols, plain_cols = _partition_object_columns(df)
    rendered_top10_cols = _render_top10_distribution_tables(df, checklist)
    plain_cols = [col for col in plain_cols if col not in rendered_top10_cols]

    # 시계열 후보(event_time 등): datetime 파싱 성공 시 일별 이벤트 수 라인 차트,
    # 실패 시 plain_cols로 되돌려 일반 막대그래프 처리
    for col in datetime_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().any():
            checklist.start(f"`{col}` 시계열 (일별 이벤트 수)")
            st.subheader(f"`{col}` 시계열 (일별 이벤트 수)")
            daily = parsed.dropna().dt.floor("D").value_counts().sort_index()
            fig = px.line(x=daily.index, y=daily.values, labels={"x": col, "y": "event count"})
            st.plotly_chart(fig, use_container_width=True)
            checklist.complete()
        else:
            if col not in rendered_top10_cols:
                plain_cols.append(col)  # 전부 NaT → 기존 막대 처리로 폴백

    # 세션 컬럼(user_session 등): 세션 해시 상위 N은 무의미 → 세션당 이벤트 수 분포(값별 개수 표)
    for col in session_cols:
        checklist.start(f"`{col}` 세션당 이벤트 수")
        st.subheader(f"`{col}` 세션당 이벤트 수 (상위 {_SESSION_PRICE_TOP_N})")
        events_per_session = df.groupby(col).size()  # NaN 세션 키는 자동 제외
        if events_per_session.empty:
            st.info("세션 데이터가 없습니다.")
            checklist.complete()
            continue
        table = _value_counts_table(events_per_session.value_counts(), "세션당 이벤트 수", top_n=_SESSION_PRICE_TOP_N)
        _render_centered_table(table)
        checklist.complete()

    # 일반 object 컬럼(event_type / category_code / brand 등): 상위 10개 값 분포 막대
    for col in plain_cols[:5]:
        checklist.start(f"`{col}` 값 분포")
        st.subheader(f"`{col}` 값 분포 (상위 10)")
        counts = df[col].value_counts().head(10)
        _render_centered_table(_value_counts_table(counts, col))
        checklist.complete()

    # 연속형 수치 컬럼: 값별 개수 표 (price 등). 고유값이 많아 상위 N개만 표시.
    for col in continuous_cols:
        checklist.start(f"`{col}` 값별 개수")
        st.subheader(f"`{col}` 값별 개수 (상위 {_SESSION_PRICE_TOP_N})")
        counts = df[col].value_counts().head(_SESSION_PRICE_TOP_N)
        table = _value_counts_table(counts, col)
        _render_centered_table(table)
        checklist.complete()

    # 범주형 ID 컬럼: 고유값이 적어 분포 막대그래프가 유의미 (category_id 등)
    for col in categorical_id_cols:
        checklist.start(f"`{col}` 값 분포")
        st.subheader(f"`{col}` 값 분포 (상위 10)")
        counts = df[col].value_counts().head(10)
        labels = [_format_id_value_for_display(value) for value in counts.index]
        fig = px.bar(x=labels, y=counts.values, labels={"x": col, "y": "count"})
        st.plotly_chart(fig, use_container_width=True)
        checklist.complete()

    # 식별자 ID 컬럼: 고유값이 많아 분포가 무의미, 고유값 개수만 요약 (product_id, user_id 등)
    if identifier_id_cols:
        checklist.start("ID 컬럼 요약 (고유값 개수)")
        summary = pd.DataFrame(
            {
                "고유값 개수": [df[c].nunique(dropna=True) for c in identifier_id_cols],
                "결측 수": [int(df[c].isnull().sum()) for c in identifier_id_cols],
            },
            index=identifier_id_cols,
        )
        st.subheader("ID 컬럼 요약")
        st.caption("product_id·user_id 등 식별자 컬럼은 히스토그램 대신 고유값 개수로 요약합니다.")
        _render_centered_dataframe(summary)
        checklist.complete()


# 추천 대회용 요약에서 쓸 컬럼명 — 데이터 개요(event_type/purchase 기반 NDCG@10)에 맞춤.
# 컬럼이 없으면 해당 항목만 조용히 건너뛴다(다른 데이터셋에도 안전).
# 제출이 유저당 Top10이므로 추천 요약의 인기·카테고리·브랜드도 Top10으로 통일.
_REC_TOP_N = 10


def _recsys_competition_summary(df: pd.DataFrame, checklist: Checklist) -> None:
    """NDCG@10(구매 예측) 대회를 겨냥한 요약. 단변량 통계로는 안 나오는 교차/집계만.

    정답은 event_type=='purchase'. purchase 행이 없으면 섹션 전체를 건너뛴다.
    인기 Top-N은 그대로 베이스라인 추천 후보가 되고, 콜드 유저·재구매율은
    폴백 전략 규모를 알려준다. 집계가 가벼워 pandas로만 처리(GPU 불필요).
    """
    if "event_type" not in df.columns:
        return  # 이 데이터셋엔 해당 없음

    checklist.start("추천 대회 요약 (NDCG@10)")
    st.header("🏆 추천 대회 요약 (NDCG@10 · 구매 예측)")

    # 1) event_type 분포 + 구매 전환율 — 정답(purchase)이 얼마나 희소한지부터.
    st.subheader("이벤트 타입 분포 · 구매 전환율")
    et = df["event_type"].value_counts(dropna=False)
    _render_centered_table(_value_counts_table(et, "event_type", top_n=10))
    views, purchases = int(et.get("view", 0)), int(et.get("purchase", 0))
    if views:
        st.caption(f"구매 전환율 ≈ {purchases / views * 100:.4f}% (purchase {purchases:,} / view {views:,})")

    purchases_df = df[df["event_type"] == "purchase"]
    if purchases_df.empty:
        st.warning("purchase 이벤트가 없어 구매 기반 요약을 건너뜁니다.")
        checklist.complete()
        return

    # 2) 구매 인기 Top-N 아이템 = 인기 베이스라인 추천 후보(전원 동일 추천 시 NDCG 바닥선).
    if "item_id" in purchases_df.columns:
        st.subheader(f"구매 인기 아이템 Top {_REC_TOP_N} (인기 베이스라인)")
        st.caption("이 Top10을 전원에게 추천한 점수가 모델이 넘어야 할 바닥선입니다.")
        _render_centered_table(_value_counts_table(purchases_df["item_id"].value_counts(), "item_id", top_n=_REC_TOP_N))

    # 3) 유저당 구매 수 분포 + 콜드(구매 0회) 유저 비율 — 폴백 대상 규모.
    if "user_id" in df.columns:
        st.subheader("유저당 구매 수 분포 · 콜드 유저 비율")
        per_user = purchases_df.groupby("user_id").size()
        total_users = df["user_id"].nunique()
        cold = total_users - per_user.shape[0]
        st.caption(
            f"전체 유저 {total_users:,}명 중 구매 이력 있는 유저 {per_user.shape[0]:,}명, "
            f"콜드 유저(구매 0회) {cold:,}명 ({cold / total_users * 100:.1f}%) → 인기 추천으로 폴백 대상."
        )
        _render_centered_table(_value_counts_table(per_user.value_counts(), "유저당 구매 수", top_n=10))

        # 4) 재구매율 — 같은 유저가 같은 아이템을 다시 사는 비율(과거 구매 = 강한 후보인지).
        if "item_id" in purchases_df.columns:
            pair_counts = purchases_df.groupby(["user_id", "item_id"]).size()
            repeat_pairs = int((pair_counts >= 2).sum())
            total_pairs = int(pair_counts.shape[0])
            if total_pairs:
                st.caption(
                    f"재구매(같은 유저·같은 아이템 2회+) 쌍 {repeat_pairs:,} / 전체 구매 쌍 {total_pairs:,} "
                    f"({repeat_pairs / total_pairs * 100:.2f}%)."
                )

    # 5) 구매 카테고리·브랜드 Top-N — 개인화 후보를 좁히는 신호.
    for col in ("category_code", "brand"):
        if col in purchases_df.columns:
            st.subheader(f"구매 {col} Top {_REC_TOP_N}")
            _render_centered_table(_value_counts_table(purchases_df[col].value_counts(dropna=False), col, top_n=_REC_TOP_N))

    # 6) 구매 시간 분포 — recency 효과/시간 분할 구조 파악.
    if "event_time" in purchases_df.columns:
        parsed = pd.to_datetime(purchases_df["event_time"], errors="coerce")
        if parsed.notna().any():
            st.subheader("구매 일별 추이")
            daily = parsed.dropna().dt.floor("D").value_counts().sort_index()
            import plotly.express as px
            fig = px.line(x=daily.index, y=daily.values, labels={"x": "날짜", "y": "구매 수"})
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    checklist.complete()


def generate_eda(df: pd.DataFrame, checklist: Checklist) -> None:
    _inject_centered_table_css()
    backend = detect_compute_backend()

    _recsys_competition_summary(df, checklist)

    with st.sidebar:
        label = "GPU: RTX 3090" if backend == "gpu" else "CPU"
        st.info(f"연산 백엔드: **{label}**")

    if backend == "gpu":
        gpu_container = st.container()
        try:
            with gpu_container:
                _eda_gpu(df, checklist)
        except Exception as e:
            gpu_container.empty()
            st.warning(f"GPU EDA 실패, CPU로 전환합니다: {e}")
            _eda_cpu(df, checklist)
    else:
        _eda_cpu(df, checklist)
