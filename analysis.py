
import pandas as pd
import plotly.express as px
import streamlit as st


def show_analytics(records):
    st.subheader("Waste Classification Analytics")

    if not records:
        st.info(
            "Your analytics will appear after you classify some images."
        )
        return

    df = pd.DataFrame(records)

    total = len(df)
    categories = df["category"].nunique()
    average_confidence = df["confidence"].mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Images Classified", total)
    col2.metric("Categories Found", categories)
    col3.metric("Mean Neighbor Vote", f"{average_confidence:.1f}%")

    counts = (
        df["category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )

    fig = px.bar(
        counts,
        x="category",
        y="count",
        title="Predictions by Waste Category",
        labels={"category": "Waste Category", "count": "Predictions"},
    )
    st.plotly_chart(fig, use_container_width=True)

    confidence_fig = px.histogram(
        df,
        x="confidence",
        nbins=10,
        title="Distribution of Neighbor-Vote Scores",
        labels={"confidence": "Neighbor-vote score"},
    )
    st.plotly_chart(confidence_fig, use_container_width=True)

    st.subheader("Prediction History")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download prediction history (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="Trashify_predictions.csv",
        mime="text/csv",
    )




#These analytics describe the images classified in the current app session. They are not a measurement of real-world waste generation or recycling impact.