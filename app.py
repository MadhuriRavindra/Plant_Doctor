
import os
import json
from collections import defaultdict
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from fastembed import ImageEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from groq import Groq

load_dotenv()

# Configuration
COLLECTION = "plant_diseases_clip"
K = 10
MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"
def get_setting(name):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        return None

LLM_MODEL = get_setting("LLM_MODEL")

DB_PATH = Path(__file__).parent / "qdrant_db"
SOLUTIONS_PATH = Path(__file__).parent / "solutions.json"

st.set_page_config(
    page_title="Plant Doctor",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Plant Rescue Assistant")
st.write(
    "Upload a photo of an affected leaf to explore possible diseases "
    "and get practical plant-care guidance."
)

# Load local resources once per Streamlit process.
@st.cache_resource
def load_resources():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "The Qdrant database is missing. Include qdrant_db "
            "in the deployment or configure a hosted Qdrant instance."
        )

    embedder = ImageEmbedding(MODEL_NAME)
    client = QdrantClient(path=str(DB_PATH))

    collection_names = [
        collection.name
        for collection in client.get_collections().collections
    ]
    if COLLECTION not in collection_names:
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION}' was not found."
        )

    with SOLUTIONS_PATH.open(encoding="utf-8") as f:
        solutions = json.load(f)

    return embedder, client, solutions


def diagnose(image, plant_choice, embedder, client):
    vector = list(embedder.embed([image]))[0].tolist()

    query_filter = None
    if plant_choice != "Not sure":
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="plant",
                    match=MatchValue(value=plant_choice),
                )
            ]
        )

    result = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=K,
        query_filter=query_filter,
        with_payload=True,
    )

    # Combine similarity scores for each plant/disease pair.
    scores = defaultdict(float)

    for point in result.points:
        payload = point.payload or {}
        plant = payload.get("plant")
        disease = payload.get("disease")

        if plant and disease and point.score is not None:
            scores[(plant, disease)] += max(0.0, point.score)

    if not scores:
        return []

    total = sum(scores.values())
    if total <= 0:
        return []

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    return [
        (plant, disease, score / total)
        for (plant, disease), score in ranked
    ]


def generate_rescue_plan(
    plant_choice, question, top_matches, solutions, groq_client
):
    context_parts = []

    for plant, disease, share in top_matches:
        details = solutions.get(f"{plant}|{disease}", {})
        context_parts.append(
            f"{plant} - {disease} "
            f"(relative match share: {share:.0%})\n"
            f"Symptoms: {details.get('symptoms', 'Not available')}\n"
            f"Treatment: {details.get('solution', 'Not available')}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a careful plant-care assistant.

The user uploaded a plant leaf photo.
Selected plant: {plant_choice}
User question: {question}

Image-similarity candidates from the local reference index:
{context}

Provide:
1. The most likely issue, clearly labelled as an estimate.
2. A practical, step-by-step care plan based on the supplied information.
3. Prevention tips.
4. How the leading possibilities may differ, if useful.

Use only the supplied candidate information for disease-specific
claims. Do not invent treatments or chemical dosages. If the evidence
is insufficient, say so and suggest getting a clearer photo or advice
from a local agricultural expert.

Image-similarity scores are not calibrated probabilities.
Do not present the relative match share as diagnostic certainty.
Recommend following product labels and keeping pesticides away
from children and pets.
"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


# Load resources and report setup errors clearly.
try:
    with st.spinner("Loading image-search model and database..."):
        embedder, qdrant, solutions = load_resources()
except Exception as exc:
    st.error(f"Could not initialize the application: {exc}")
    st.info(
        "Check the Qdrant database, installed dependencies, "
        "and model download access."
    )
    st.stop()

# Groq configuration
api_key = get_setting("GROQ_API_KEY")

if not api_key or not LLM_MODEL:
    st.error(
        "Groq configuration is missing. Set GROQ_API_KEY and "
        "LLM_MODEL in your local .env file or Streamlit Cloud Secrets."
    )
    st.stop()

groq_client = Groq(api_key=api_key)

# Single diagnosis interface
with st.sidebar:
    st.header("Check your plant")

    plant_choice = st.selectbox(
        "Which plant is affected?",
        ["Not sure", "Tomato", "Potato", "Pepper bell"],
    )

    uploaded_file = st.file_uploader(
        "Upload a clear leaf photo",
        type=["jpg", "jpeg", "png"],
    )

    question = st.text_input(
        "What would you like to know?",
        value="How can I help this plant recover?",
    )

    diagnose_clicked = st.button(
        "Diagnose plant",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

if uploaded_file is None:
    st.info(
        "Choose a plant (or select 'Not sure') and upload a photo "
        "of one affected leaf to begin."
    )
    st.caption(
        "Results are image-based suggestions, not a confirmed diagnosis."
    )

elif diagnose_clicked:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.image(
                image,
                caption="Uploaded leaf photo",
                use_container_width=True,
            )

        with st.spinner("Searching similar plant images..."):
            matches = diagnose(
                image, plant_choice, embedder, qdrant
            )

        with right:
            st.subheader("Possible matches")

            if not matches:
                st.warning(
                    "No useful matches were found. Try a clearer "
                    "photo with one leaf filling most of the frame."
                )
                st.stop()

            for rank, (plant, disease, share) in enumerate(
                matches, start=1
            ):
                st.markdown(
                    f"**{rank}. {plant} — {disease}**  \n"
                    f"Relative match share: {share:.0%}"
                )

            if matches[0][2] < 0.5:
                st.warning(
                    "The leading match is not strong enough to rely "
                    "on. Treat this result cautiously."
                )

            if matches[0][1].lower() == "healthy":
                st.info(
                    "The closest match is healthy. This does not "
                    "rule out a disease; monitor the plant and "
                    "consider another photo if symptoms persist."
                )

        with st.spinner("Preparing plant-care guidance..."):
            advice = generate_rescue_plan(
                plant_choice,
                question,
                matches,
                solutions,
                groq_client,
            )

        st.divider()
        st.subheader("🌿 Suggested rescue plan")
        st.markdown(advice)

        st.caption(
            "This is an AI-generated, image-based estimate. "
            "Check the plant and local growing conditions before "
            "taking action."
        )

    except UnidentifiedImageError:
        st.error("This image could not be opened. Please upload a valid photo.")
    except Exception as exc:
        st.error(f"Diagnosis failed: {exc}")
        st.caption(
            "Check the model, Qdrant database, API configuration, "
            "and application logs."
        )
