
import streamlit as st
from PIL import Image

from classifier import DATA_DIR, predict_image, train_classifier
from recommendations import get_recommendation
from analytics import show_analytics


st.set_page_config(
    page_title="Trashify AI",
    page_icon="♻️",
    layout="wide",
)

st.title("♻️ Trashify AI")
st.caption(
    "An experimental waste classification and recycling analytics project"
)

st.markdown(
    """
    Upload a waste image, explore the model's prediction, and review
    general recycling guidance. Predictions may be incorrect.
    """
)


@st.cache_resource
def get_model():
    return train_classifier()


with st.sidebar:
    st.header("About the model")
    st.write("Algorithm: K-nearest neighbors (KNN)")
    st.write("Features: RGB histograms and spatial color averages")
    st.write("Training source: Your local labeled image folders")

    st.warning(
        "This baseline uses simple visual features. "
        "Do not use it for safety-critical decisions."
    )


tab_classify, tab_analytics, tab_guide = st.tabs(
    ["Classify Waste", "Analytics", "Recycling Guide"]
)


with tab_classify:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded image",
                width=350,
            )

            if st.button("Classify image", type="primary"):
                with st.spinner("Loading the trained model..."):
                    model, sample_count, trained_classes = get_model()

                if len(trained_classes) < 2:
                    st.error("Train with at least two waste categories.")
                else:
                    category, ranking = predict_image(image, model)

                    # Session history is intentionally not permanent.
                    record = {
                        "category": category,
                        "confidence": ranking[0]["confidence"],
                        "training_images": sample_count,
                    }

                    if "history" not in st.session_state:
                        st.session_state.history = []

                    st.session_state.history.append(record)

                    st.success(f"Predicted category: {category.title()}")

                    st.caption(
                        "The score below represents KNN neighbor votes, "
                        "not a calibrated probability of correctness."
                    )

                    st.progress(ranking[0]["confidence"])
                    st.write(
                        "Top neighbor-vote score: "
                        f"{ranking[0]['confidence']:.1%}"
                    )

                    st.subheader("Category scores")
                    for item in ranking:
                        st.write(
                            f"{item['category'].title()}: "
                            f"{item['confidence']:.1%}"
                        )

                    guidance = get_recommendation(category)
                    st.subheader("Recycling guidance")
                    st.write(
                        f"**Recyclability:** {guidance['recyclable']}"
                    )
                    st.write(guidance["tip"])
                    st.info(guidance["caution"])

        except (OSError, ValueError) as exc:
            st.error(f"Could not process this image: {exc}")

        except Exception as exc:
            st.error(f"Classification failed: {exc}")

    else:
        st.info(
            "Upload an image to begin. The classifier requires a labeled "
            "training dataset in the data folder."
        )


with tab_analytics:
    show_analytics(st.session_state.get("history", []))


with tab_guide:
    st.subheader("Waste sorting guide")

    for category in [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    ]:
        guidance = get_recommendation(category)

        with st.expander(category.title()):
            st.write(f"**Recyclability:** {guidance['recyclable']}")
            st.write(guidance["tip"])
            st.caption(guidance["caution"])



import streamlit as st
from PIL import Image

from classifier import predict_image, train_classifier
from recommendation import get_recommendation
from analytics import show_analytics

st.set_page_config(
    page_title="Trashify",
    page_icon="♻️",
    layout="wide",
)

st.title("♻️ Trashify")
st.subheader("AI-Powered Waste Classification")

st.write(
    "Upload an image of cardboard, glass, metal, paper, plastic, or trash."
)

@st.cache_resource
def load_model():
    return train_classifier()

classify_tab, analytics_tab, guide_tab = st.tabs(
    ["Classify Waste", "Analytics", "Recycling Guide"]
)

with classify_tab:
    uploaded = st.file_uploader(
        "Upload waste image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, width=350)

        if st.button("Classify Image"):
            try:
                model, count, classes = load_model()
                category, ranking = predict_image(image, model)

                st.success(f"Predicted category: {category.title()}")
                st.write(f"Training images: {count}")

                for item in ranking:
                    st.write(
                        f"{item['category'].title()}: "
                        f"{item['confidence']:.1%}"
                    )

                advice = get_recommendation(category)
                st.subheader("Recycling Recommendation")
                st.write(advice["recyclable"])
                st.write(advice["tip"])
                st.info(advice["caution"])

                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    "category": category,
                    "confidence": ranking[0]["confidence"],
                })

            except Exception as error:
                st.error(str(error))

with analytics_tab:
    show_analytics(st.session_state.get("history", []))

with guide_tab:
    for category in [
        "cardboard", "glass", "metal",
        "paper", "plastic", "trash"
    ]:
        with st.expander(category.title()):
            st.write(get_recommendation(category)["tip"])
