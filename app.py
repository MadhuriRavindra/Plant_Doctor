import os, json
from collections import defaultdict
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from fastembed import ImageEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from groq import Groq

load_dotenv()

COLLECTION = "plant_diseases_clip"
K = 10
LLM_MODEL = os.getenv("LLM_MODEL")

st.set_page_config(page_title="Plant Doctor", layout="wide")

# --- Session state: a counter changes the uploader's key, which resets it ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "history" not in st.session_state:
    st.session_state.history = []   # list of (image, result) for past checks

def diagnose(image: Image.Image):
    # TODO: call your existing code here (fastembed -> Qdrant -> solution)
    # return the top label and the solution text
    return "Tomato Early blight", "Remove affected leaves, avoid overhead watering..."

# ---------------- Sidebar: upload + controls ----------------
with st.sidebar:
    st.header("Check a plant")
    uploaded = st.file_uploader(
        "Upload a leaf photo",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    question = st.text_input("Question (optional)", "How do I save this plant?")
    run = st.button("Diagnose", type="primary", use_container_width=True, disabled=uploaded is None)

    if st.button("Clear / new image", use_container_width=True):
        st.session_state.uploader_key += 1   # new key = empty uploader
        st.rerun()

# ---------------- Main area: result ----------------
st.title("Plant Disease Diagnosis")

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded image", width="stretch")
        st.button("Diagnose", type="primary", width="stretch", disabled=uploaded is None)
        st.button("Clear / new image", width="stretch")
    with col2:
        if run:
            with st.spinner("Analyzing..."):
                label, solution = diagnose(img)
            st.session_state.history.insert(0, (img, label, solution))
            st.subheader(label)
            st.write(solution)
else:
    st.info("Use the sidebar to upload a leaf photo.")

# ---------------- Past checks ----------------
if st.session_state.history:
    st.divider()
    st.subheader("Previous checks")
    for i, (im, label, sol) in enumerate(st.session_state.history[:5]):
        with st.expander(f"{label}", expanded=False):
            st.image(im, width=200)
            st.write(sol)

@st.cache_resource
def load():
    return ImageEmbedding("Qdrant/clip-ViT-B-32-vision"), QdrantClient(path="qdrant_db")

embedder, qdrant = load()
solutions = json.load(open("solutions.json", encoding="utf-8"))
groq = Groq(api_key=os.environ["GROQ_API_KEY"])

def diagnose(img, plant_choice):
    vec = list(embedder.embed([img]))[0].tolist()
    flt = None
    if plant_choice != "Not sure":
        flt = Filter(must=[FieldCondition(key="plant", match=MatchValue(value=plant_choice))])
    hits = qdrant.query_points(COLLECTION, query=vec, limit=K, query_filter=flt).points
    scores = defaultdict(float)
    for h in hits:
        scores[(h.payload["plant"], h.payload["disease"])] += h.score
    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:3]
    return [(p, d, s / total) for (p, d), s in ranked]

st.title("Plant Rescue Assistant")

if not LLM_MODEL:
    st.error("LLM_MODEL is not set. Add it to your .env file (run list_models.py to see valid ids).")
    st.stop()

plant_choice = st.selectbox("Which plant is it?", ["Not sure", "Tomato", "Potato", "Pepper bell"])
file = st.file_uploader("Upload a photo of the affected leaf", type=["jpg", "jpeg", "png"])
question = st.text_input("Your question", "How can I save this plant?")

if file and st.button("Diagnose"):
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)
    top3 = diagnose(img, plant_choice)

    st.subheader("Most likely matches")
    for plant, disease, share in top3:
        st.write(f"**{plant} - {disease}**: {share:.0%} of matches")

    if top3[0][2] < 0.5:
        st.warning("Low confidence. Please upload a clearer, close-up photo of a single leaf.")
    if top3[0][1] == "healthy" and top3[0][2] < 0.8:
        st.warning("Mostly looks healthy, but some matches suggest early disease. "
                   "Re-check in a few days with a closer photo.")

    context = "\n\n".join(
        f"{plant} - {disease} ({share:.0%} match)\n"
        f"Symptoms: {solutions.get(f'{plant}|{disease}', {}).get('symptoms', 'n/a')}\n"
        f"Treatment: {solutions.get(f'{plant}|{disease}', {}).get('solution', 'n/a')}"
        for plant, disease, share in top3
    )

    prompt = f"""You are a careful plant-care assistant. A user uploaded a plant photo.
The user says the plant is: {plant_choice}. If this is "Not sure", mention that the plant species may be misidentified.
Question: {question}

Candidate diagnoses from an image-similarity search (with match share):
{context}

Write: 1) the most likely problem, 2) step-by-step rescue plan, 3) prevention tips,
4) one line on how to tell it apart from the runner-up if the top two are close.
Use only the information above. Say clearly that this is an image-based estimate, and
advise following product labels and keeping chemicals away from children and pets."""

    with st.spinner("Writing advice..."):
        resp = groq.chat.completions.create(
            model=LLM_MODEL, messages=[{"role": "user", "content": prompt}]
        )
    st.subheader("Rescue plan")
    st.write(resp.choices[0].message.content)