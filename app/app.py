import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from treatment import get_advice  # noqa: E402

st.set_page_config(page_title="AgriSense AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {background: linear-gradient(135deg,#1b5e20 0%,#43a047 60%,#9ccc65 100%);
       padding: 3rem 2.5rem; border-radius: 22px; color: white; margin-bottom: 1.5rem;
       box-shadow: 0 10px 30px rgba(46,125,50,.25);}
.hero h1 {font-size: 2.8rem; margin: 0 0 .4rem 0; color: white;}
.hero p {font-size: 1.15rem; opacity: .95; margin: 0; max-width: 700px;}
.card {background: white; border-radius: 16px; padding: 1.4rem; height: 100%;
       box-shadow: 0 4px 18px rgba(0,0,0,.07); border: 1px solid #e3eedd;}
.card h3 {margin-top: 0; color: #1b5e20;}
.stat {background: white; border-radius: 16px; padding: 1.2rem; text-align: center;
       box-shadow: 0 4px 18px rgba(0,0,0,.07); border: 1px solid #e3eedd;}
.stat .num {font-size: 2.1rem; font-weight: 800; color: #2e7d32;}
.stat .lbl {color: #5b6b5b; font-size: .9rem;}
.result {border-radius: 18px; padding: 1.5rem; color: white; margin-bottom: 1rem;}
.result.ok {background: linear-gradient(135deg,#2e7d32,#66bb6a);}
.result.bad {background: linear-gradient(135deg,#c62828,#ef6c00);}
.result.warn {background: linear-gradient(135deg,#616161,#9e9e9e);}
.result h2 {margin: 0; color: white;}
.result .sub {opacity: .9; font-size: 1rem;}
.badge {display:inline-block; padding:.2rem .8rem; border-radius:999px;
        background: rgba(255,255,255,.25); font-weight:700; margin-bottom:.5rem;}
.bar-row {margin: .7rem 0;}
.bar-label {display:flex; justify-content:space-between; font-size:.95rem; margin-bottom:.2rem;}
.bar-bg {background:#e4ece0; border-radius:99px; height:12px;}
.bar-fg {height:12px; border-radius:99px;}
.soon {opacity: .85; border-style: dashed;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(ROOT / "models" / "leaf_model.keras")


@st.cache_data
def load_labels():
    names = json.loads((ROOT / "models" / "class_names.json").read_text())
    info = json.loads((ROOT / "models" / "class_info.json").read_text())
    return names, info


def predict(img):
    names, _ = load_labels()
    arr = np.array(img.convert("RGB").resize((224, 224)), dtype="float32")[None, ...]
    probs = load_model().predict(arr, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    return [(names[i], float(probs[i])) for i in top]


def bar(label, p, top=False):
    color = "#2e7d32" if top else "#a5c9a1"
    return (f"<div class='bar-row'><div class='bar-label'><span>{label}</span>"
            f"<b>{p*100:.1f}%</b></div><div class='bar-bg'>"
            f"<div class='bar-fg' style='width:{p*100:.1f}%;background:{color}'></div></div></div>")


st.sidebar.markdown("## 🌿 AgriSense AI")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔬 Disease Detection",
                                      "📊 Model Performance", "🚀 Roadmap"],
                        label_visibility="collapsed")
st.sidebar.caption("3rd-year project: Smart Agriculture using AI")

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.markdown("""<div class='hero'><h1>🌿 AgriSense AI</h1>
    <p>An AI-powered assistant that helps farmers detect crop leaf diseases early,
    get treatment guidance, and make smarter decisions.</p></div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [(c1, "97%", "Test accuracy"), (c2, "38", "Disease classes"),
                          (c3, "54,305", "Training images"), (c4, "14", "Crops covered")]:
        col.markdown(f"<div class='stat'><div class='num'>{num}</div>"
                     f"<div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    st.write("")
    a, b, c = st.columns(3)
    a.markdown("<div class='card'><h3>🔬 Disease Detection</h3>Upload a leaf photo and "
               "get the disease, confidence, and treatment advice in seconds.</div>",
               unsafe_allow_html=True)
    b.markdown("<div class='card'><h3>📊 Transparent Results</h3>See accuracy, the confusion "
               "matrix, and which diseases the model finds hardest.</div>",
               unsafe_allow_html=True)
    c.markdown("<div class='card'><h3>🚀 Growing Platform</h3>Yield prediction, irrigation "
               "advice, and weather forecasts are coming next.</div>", unsafe_allow_html=True)

    st.write("")
    st.info("👈 Open **Disease Detection** in the sidebar to try it with your own leaf photo.")

# ---------------- DETECTION ----------------
elif page == "🔬 Disease Detection":
    st.title("🔬 Leaf Disease Detection")
    st.caption("Upload a clear, close photo of a single leaf. Works best with one leaf "
               "against a plain background.")

    threshold = st.slider("Minimum confidence to trust a result", 0.5, 0.95, 0.70, 0.05)
    file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

    if file:
        img = Image.open(file)
        left, right = st.columns([1, 1.3], gap="large")
        left.image(img, caption="Uploaded leaf", use_container_width=True)

        with st.spinner("Analysing leaf..."):
            results = predict(img)

        _, info = load_labels()
        best, conf = results[0]
        meta = info[best]
        title = f"{meta['crop']}: {meta['disease']}"

        with right:
            if conf < threshold:
                st.markdown(f"<div class='result warn'><span class='badge'>LOW CONFIDENCE</span>"
                            f"<h2>Not sure about this one</h2><div class='sub'>Best guess: {title} "
                            f"({conf*100:.0f}%). Try a clearer, closer photo of one leaf.</div></div>",
                            unsafe_allow_html=True)
            else:
                cls = "ok" if meta["healthy"] else "bad"
                badge = "HEALTHY" if meta["healthy"] else "DISEASE DETECTED"
                st.markdown(f"<div class='result {cls}'><span class='badge'>{badge}</span>"
                            f"<h2>{title}</h2><div class='sub'>Confidence: {conf*100:.1f}%</div></div>",
                            unsafe_allow_html=True)

            st.markdown("**Top 3 predictions**")
            html = ""
            for i, (name, p) in enumerate(results):
                m = info[name]
                html += bar(f"{m['crop']}: {m['disease']}", p, top=(i == 0))
            st.markdown(html, unsafe_allow_html=True)

        if conf >= threshold:
            adv = get_advice(best)
            st.subheader("🩺 About & Recommended Actions")
            st.markdown(f"<div class='card'><b>About:</b> {adv['about']}<br><br><b>What to do:</b><ul>"
                        + "".join(f"<li>{a}</li>" for a in adv["action"])
                        + "</ul></div>", unsafe_allow_html=True)
        st.caption("AI suggestion only. Please confirm with a local agriculture expert before "
                   "applying any treatment.")

# ---------------- PERFORMANCE ----------------
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.write("Model: **MobileNetV2** (transfer learning, 2-phase fine-tuning) trained on the "
             "PlantVillage dataset with a 70/15/15 train/validation/test split.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", "97%")
    c2.metric("Macro F1-score", "0.96")
    c3.metric("Test images", "8,146")
    c4.metric("Classes", "38")

    st.subheader("Training curves")
    p = ROOT / "docs" / "training_curves.png"
    if p.exists():
        st.image(str(p), use_container_width=True)

    st.subheader("Weakest classes")
    st.dataframe([
        {"Class": "Tomato: Early blight", "Recall": 0.62, "F1": 0.75, "Why": "Looks like Target Spot / Septoria"},
        {"Class": "Potato: Healthy", "Recall": 0.74, "F1": 0.83, "Why": "Only 23 test images"},
        {"Class": "Tomato: Target Spot", "Recall": 0.91, "F1": 0.84, "Why": "Confused with Early blight"},
        {"Class": "Corn: Gray leaf spot", "Recall": 0.90, "F1": 0.88, "Why": "Similar to Northern Leaf Blight"},
        {"Class": "Tomato: Mosaic virus", "Recall": 0.82, "F1": 0.89, "Why": "Small class (56 images)"},
    ], use_container_width=True, hide_index=True)

    st.subheader("Confusion matrix")
    p = ROOT / "docs" / "confusion_matrix.png"
    if p.exists():
        st.image(str(p), use_container_width=True)

    p = ROOT / "docs" / "classification_report.txt"
    if p.exists():
        with st.expander("Full classification report"):
            st.code(p.read_text())

    st.warning("PlantVillage photos have plain backgrounds. Accuracy on real field photos "
               "is usually lower, which is why the app shows a confidence score and "
               "asks for a clearer photo when unsure.")

# ---------------- ROADMAP ----------------
else:
    st.title("🚀 Roadmap")
    st.write("What's built and what's coming next.")
    st.success("✅ **Done:** Leaf disease detection (97% test accuracy, 38 classes)")
    a, b, c = st.columns(3)
    a.markdown("<div class='card soon'><h3>🌾 Yield Prediction</h3>Predict yield per hectare "
               "from crop, state, season, area and rainfall.<br><br><b>Status: in progress</b></div>",
               unsafe_allow_html=True)
    b.markdown("<div class='card soon'><h3>💧 Irrigation Advice</h3>Daily water requirement "
               "using evapotranspiration and crop stage.<br><br><b>Status: planned</b></div>",
               unsafe_allow_html=True)
    c.markdown("<div class='card soon'><h3>⛅ Weather</h3>Live conditions and 7-day forecast "
               "for the farmer's location.<br><br><b>Status: planned</b></div>",
               unsafe_allow_html=True)