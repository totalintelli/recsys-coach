import pandas as pd
import streamlit as st


def detect_compute_backend() -> str:
    """'gpu' 또는 'cpu' 반환"""
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
    except ImportError:
        pass
    return "cpu"


def _eda_gpu(df: pd.DataFrame) -> None:
    import cudf
    import plotly.express as px

    gdf = cudf.DataFrame.from_pandas(df)

    st.subheader("기초 통계")
    st.dataframe(gdf.describe().to_pandas())

    num_cols = df.select_dtypes("number").columns.tolist()
    if len(num_cols) >= 2:
        st.subheader("상관관계 히트맵")
        corr = gdf[num_cols].corr().to_pandas()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("결측값")
    missing = gdf.isnull().sum().to_pandas().rename("결측 수")
    st.dataframe(missing[missing > 0])

    for col in df.select_dtypes("object").columns[:5]:
        st.subheader(f"`{col}` 값 분포 (상위 10)")
        counts = gdf[col].value_counts().head(10).to_pandas()
        fig = px.bar(counts, x=counts.index, y=counts.values, labels={"x": col, "y": "count"})
        st.plotly_chart(fig, use_container_width=True)


def _eda_cpu(df: pd.DataFrame) -> None:
    from streamlit_pandas_profiling import st_profile_report
    from ydata_profiling import ProfileReport

    report = ProfileReport(df, minimal=False, progress_bar=False)
    st_profile_report(report)


def generate_eda(df: pd.DataFrame) -> None:
    backend = detect_compute_backend()

    with st.sidebar:
        label = "GPU: RTX 3090" if backend == "gpu" else "CPU"
        st.info(f"연산 백엔드: **{label}**")

    if backend == "gpu":
        try:
            _eda_gpu(df)
        except Exception as e:
            st.warning(f"GPU EDA 실패, CPU로 전환합니다: {e}")
            _eda_cpu(df)
    else:
        _eda_cpu(df)
