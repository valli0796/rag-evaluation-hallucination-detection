
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

REPORT_PATH = Path("reports/experiment_results.csv")

st.title("RAG Evaluation & Hallucination Detection")
st.caption("Compare retrieval configurations and evaluate answer evidence support.")

if not REPORT_PATH.exists():
    st.error("Experiment report not found. Run the experiment runner first.")
    st.stop()

df = pd.read_csv(REPORT_PATH)

required_columns = {
    "experiment",
    "question",
    "chunk_size",
    "query_time_seconds",
    "evidence_support_percentage",
    "supported_statements",
    "unsupported_statements"
}

missing = required_columns - set(df.columns)

if missing:
    st.error(f"Missing columns in report: {', '.join(sorted(missing))}")
    st.stop()

st.sidebar.header("Dashboard Filters")

experiments = sorted(df["experiment"].dropna().unique())

selected_experiments = st.sidebar.multiselect(
    "Select experiments",
    experiments,
    default=experiments
)

filtered_df = df[
    df["experiment"].isin(selected_experiments)
].copy()

if filtered_df.empty:
    st.warning("Select at least one experiment.")
    st.stop()

summary = (
    filtered_df.groupby("experiment")
    .agg(
        average_support=("evidence_support_percentage", "mean"),
        average_query_time=("query_time_seconds", "mean"),
        supported_statements=("supported_statements", "sum"),
        unsupported_statements=("unsupported_statements", "sum"),
        questions_evaluated=("question", "count")
    )
    .reset_index()
)

average_support = filtered_df[
    "evidence_support_percentage"
].mean()

average_time = filtered_df[
    "query_time_seconds"
].mean()

total_supported = filtered_df[
    "supported_statements"
].sum()

total_unsupported = filtered_df[
    "unsupported_statements"
].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Evidence Support",
    f"{average_support:.2f}%"
)

col2.metric(
    "Average Query Time",
    f"{average_time:.2f} sec"
)

col3.metric(
    "Supported Statements",
    int(total_supported)
)

col4.metric(
    "Unsupported Statements",
    int(total_unsupported)
)

st.divider()

st.subheader("Experiment Comparison")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_support = px.bar(
        summary,
        x="experiment",
        y="average_support",
        color="experiment",
        text_auto=".2f",
        title="Average Evidence Support",
        labels={
            "experiment": "Experiment",
            "average_support": "Evidence Support (%)"
        }
    )
    fig_support.update_layout(showlegend=False)
    st.plotly_chart(fig_support, use_container_width=True)

with chart_col2:
    fig_time = px.bar(
        summary,
        x="experiment",
        y="average_query_time",
        color="experiment",
        text_auto=".2f",
        title="Average Query Execution Time",
        labels={
            "experiment": "Experiment",
            "average_query_time": "Query Time (seconds)"
        }
    )
    fig_time.update_layout(showlegend=False)
    st.plotly_chart(fig_time, use_container_width=True)

st.divider()

st.subheader("Question-wise Evidence Support")

fig_questions = px.bar(
    filtered_df,
    x="question_number",
    y="evidence_support_percentage",
    color="experiment",
    barmode="group",
    hover_data=["question", "chunk_size"],
    labels={
        "question_number": "Question Number",
        "evidence_support_percentage": "Evidence Support (%)"
    }
)

st.plotly_chart(fig_questions, use_container_width=True)

st.divider()

st.subheader("Experiment Summary")
st.dataframe(
    summary.round(2),
    use_container_width=True,
    hide_index=True
)

st.subheader("Detailed Evaluation Results")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Filtered Results",
    data=csv_data,
    file_name="rag_evaluation_results.csv",
    mime="text/csv"
)

st.caption(
    "Note: Evidence support is calculated using keyword matching "
    "and basic negation detection. It is not a validated semantic "
    "faithfulness measurement."
)
