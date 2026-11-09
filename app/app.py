import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from treatment import get_advice  # noqa: E402

st.set_page_config(page_title="AgriSense AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: 'Poppins', sans-serif;}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1150px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#e6f0e1 0%,#cfe5c8 100%);}
h1, h2, h3 {color: #1b5e20;}

.hero {background: linear-gradient(135deg,#1b5e20 0%,#43a047 60%,#9ccc65 100%);
       padding: 3rem 2.5rem; border-radius: 22px; color: white; margin-bottom: 1.5rem;
       box-shadow: 0 10px 30px rgba(46,125,50,.25);}
.hero h1 {font-size: 2.8rem; margin: 0 0 .4rem 0; color: white;}
.hero p {font-size: 1.15rem; opacity: .95; margin: 0; max-width: 700px;}

.card {background: white; border-radius: 16px; padding: 1.4rem; height: 100%;
       box-shadow: 0 4px 18px rgba(0,0,0,.07); border: 1px solid #e3eedd;
       transition: transform .2s, box-shadow .2s;}
.card:hover {transform: translateY(-4px); box-shadow: 0 10px 26px rgba(0,0,0,.12);}
.card h3 {margin-top: 0; color: #1b5e20;}
.done {border-left: 6px solid #2e7d32;}
.soon {border-left: 6px solid #f9a825;}

.stat {background: white; border-radius: 16px; padding: 1.2rem; text-align: center;
       box-shadow: 0 4px 18px rgba(0,0,0,.07); border: 1px solid #e3eedd;}
.stat .num {font-size: 2.1rem; font-weight: 800; color: #2e7d32;}
.stat .lbl {color: #5b6b5b; font-size: .9rem;}

.result {border-radius: 18px; padding: 1.5rem; color: white; margin-bottom: 1rem;}
.result.ok {background: linear-gradient(135deg,#2e7d32,#66bb6a);}
.result.bad {background: linear-gradient(135deg,#c62828,#ef6c00);}
.result.warn {background: linear-gradient(135deg,#616161,#9e9e9e);}
.result.blue {background: linear-gradient(135deg,#0d47a1,#42a5f5);}
.result h2 {margin: 0; color: white;}
.result .sub {opacity: .92; font-size: 1rem;}
.badge {display:inline-block; padding:.2rem .8rem; border-radius:999px;
        background: rgba(255,255,255,.25); font-weight:700; margin-bottom:.5rem;}

.bar-row {margin: .7rem 0;}
.bar-label {display:flex; justify-content:space-between; font-size:.95rem; margin-bottom:.2rem;}
.bar-bg {background:#e4ece0; border-radius:99px; height:12px;}
.bar-fg {height:12px; border-radius:99px;}

.wx-main {background: linear-gradient(135deg,#0d47a1,#1e88e5 55%,#64b5f6);
          border-radius: 22px; padding: 2rem 2.2rem; color: white;
          box-shadow: 0 10px 30px rgba(13,71,161,.25); margin-bottom: 1rem;}
.wx-main .city {font-size: 1.1rem; opacity: .9;}
.wx-main .temp {font-size: 4rem; font-weight: 800; line-height: 1.1;}
.wx-main .desc {font-size: 1.3rem;}
.wx-row {display:flex; gap:1.5rem; margin-top:1rem; flex-wrap:wrap;}
.wx-row div {background: rgba(255,255,255,.18); padding:.5rem 1rem; border-radius:12px; font-size:.95rem;}
.day {background:white; border-radius:16px; padding:1rem .5rem; text-align:center;
      border:1px solid #dbe8f5; box-shadow:0 4px 14px rgba(0,0,0,.06);}
.day .d {font-weight:700; color:#0d47a1;}
.day .e {font-size:2rem;}
.day .t {font-size:.95rem;}
.day .r {font-size:.85rem; color:#1565c0;}
.tip {background:#fff8e1; border-left:5px solid #f9a825; border-radius:10px;
      padding:.7rem 1rem; margin:.5rem 0;}
</style>
""", unsafe_allow_html=True)


# ---------------- loaders ----------------
@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(ROOT / "models" / "leaf_model.keras")


@st.cache_data
def load_labels():
    names = json.loads((ROOT / "models" / "class_names.json").read_text())
    info = json.loads((ROOT / "models" / "class_info.json").read_text())
    return names, info


@st.cache_resource
def load_yield_model():
    return joblib.load(ROOT / "models" / "yield_model.joblib")


@st.cache_data
def load_yield_options():
    return json.loads((ROOT / "models" / "yield_options.json").read_text())


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


# ---------------- weather helpers ----------------
WX = {0: ("☀️", "Clear sky"), 1: ("🌤️", "Mainly clear"), 2: ("⛅", "Partly cloudy"),
      3: ("☁️", "Overcast"), 45: ("🌫️", "Fog"), 48: ("🌫️", "Fog"),
      51: ("🌦️", "Light drizzle"), 53: ("🌦️", "Drizzle"), 55: ("🌦️", "Heavy drizzle"),
      61: ("🌧️", "Light rain"), 63: ("🌧️", "Rain"), 65: ("🌧️", "Heavy rain"),
      71: ("❄️", "Light snow"), 73: ("❄️", "Snow"), 75: ("❄️", "Heavy snow"),
      80: ("🌦️", "Showers"), 81: ("🌧️", "Heavy showers"), 82: ("⛈️", "Violent showers"),
      95: ("⛈️", "Thunderstorm"), 96: ("⛈️", "Thunderstorm, hail"), 99: ("⛈️", "Severe thunderstorm")}


def wx(code):
    return WX.get(int(code), ("🌡️", "Weather"))


@st.cache_data(ttl=3600)
def geocode(name):
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": name, "count": 6, "language": "en"}, timeout=10)
        return r.json().get("results", []) or []
    except Exception:
        return []


@st.cache_data(ttl=900)
def fetch_weather(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 7,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                   "precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "precipitation_probability_max,et0_fao_evapotranspiration",
    }, timeout=15)
    r.raise_for_status()
    return r.json()


def pick_location(key):
    q = st.text_input("Search city or village", "Delhi", key=key + "_q")
    results = geocode(q) if q else []
    if not results:
        st.warning("Location not found. Try another spelling.")
        return None
    labels = [f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}" for r in results]
    i = st.selectbox("Select location", range(len(results)),
                     format_func=lambda i: labels[i], key=key + "_s")
    return {"label": labels[i], "lat": results[i]["latitude"], "lon": results[i]["longitude"]}


# ---------------- irrigation data ----------------
# Crop coefficient Kc (FAO-56, approximate): initial, mid-season, late
KC = {
    "Wheat": (0.40, 1.15, 0.40), "Rice": (1.05, 1.20, 0.90), "Maize": (0.30, 1.20, 0.60),
    "Tomato": (0.60, 1.15, 0.80), "Potato": (0.50, 1.15, 0.75), "Cotton": (0.35, 1.15, 0.70),
    "Sugarcane": (0.40, 1.25, 0.75), "Soybean": (0.40, 1.15, 0.50), "Mustard": (0.35, 1.15, 0.35),
    "Onion": (0.70, 1.05, 0.75), "Apple": (0.50, 0.95, 0.70), "Grape": (0.30, 0.85, 0.45),
}
STAGES = ["Initial (just planted)", "Mid-season (flowering / fruiting)", "Late (ripening)"]
METHODS = {"Drip": 0.90, "Sprinkler": 0.75, "Flood / Furrow": 0.60}

# ---------------- sidebar ----------------
st.sidebar.markdown("## 🌿 AgriSense AI")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔬 Disease Detection", "🌾 Yield Prediction", "⛅ Weather",
     "💧 Irrigation Advice", "📊 Model Performance", "🚀 Roadmap"],
    label_visibility="collapsed",
)
st.sidebar.caption("3rd-year project: Smart Agriculture using AI")

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.markdown("""<div class='hero'><h1>🌿 AgriSense AI</h1>
    <p>An AI-powered assistant that helps farmers detect crop diseases, estimate yield,
    check the weather, and irrigate smarter, all in one place.</p></div>""",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [(c1, "97%", "Disease model accuracy"), (c2, "38", "Disease classes"),
                          (c3, "0.94", "Yield model R²"), (c4, "7-day", "Weather forecast")]:
        col.markdown(f"<div class='stat'><div class='num'>{num}</div>"
                     f"<div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    st.write("")
    a, b = st.columns(2)
    a.markdown("<div class='card done'><h3>🔬 Disease Detection</h3>Upload a leaf photo and get "
               "the disease, confidence, and treatment advice in seconds.</div>",
               unsafe_allow_html=True)
    b.markdown("<div class='card done'><h3>🌾 Yield Prediction</h3>Estimate tonnes per hectare "
               "for any state, district, crop and season.</div>", unsafe_allow_html=True)
    st.write("")
    c, d = st.columns(2)
    c.markdown("<div class='card done'><h3>⛅ Weather</h3>Live conditions and a 7-day forecast "
               "for any place, with farming alerts.</div>", unsafe_allow_html=True)
    d.markdown("<div class='card done'><h3>💧 Irrigation Advice</h3>Daily water need in litres, "
               "from evapotranspiration, rain and crop stage.</div>", unsafe_allow_html=True)

    st.write("")
    st.info("👈 Pick a feature from the sidebar to try it.")

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

# ---------------- YIELD ----------------
elif page == "🌾 Yield Prediction":
    st.title("🌾 Crop Yield Prediction")
    st.caption("Estimate yield per hectare from historical district-level records "
               "(India, 1997 to 2015).")

    if not (ROOT / "models" / "yield_model.joblib").exists():
        st.error("Yield model not found. Put `yield_model.joblib` and `yield_options.json` "
                 "in the `models` folder.")
        st.stop()

    opts = load_yield_options()
    c1, c2 = st.columns(2)
    state = c1.selectbox("State", opts["states"])
    district = c2.selectbox("District", opts["districts"][state])
    c3, c4 = st.columns(2)
    crop = c3.selectbox("Crop", opts["crops"])
    season = c4.selectbox("Season", opts["seasons"])
    c5, c6 = st.columns(2)
    area = c5.number_input("Area (hectares)", min_value=0.1, value=2.0, step=0.5)
    year = c6.number_input("Year", min_value=opts["year_min"], max_value=2035,
                           value=opts["year_max"], step=1)

    if st.button("Predict yield", type="primary"):
        row = pd.DataFrame([{
            "State_Name": state, "District_Name": district, "Season": season,
            "Crop": crop, "Crop_Year": int(year), "Area": float(area),
        }])
        y = float(np.expm1(load_yield_model().predict(row)[0]))
        st.markdown(f"<div class='result ok'><span class='badge'>ESTIMATE</span>"
                    f"<h2>{y:.2f} tonnes per hectare</h2>"
                    f"<div class='sub'>For {area} ha: about {y*area:.1f} tonnes total</div></div>",
                    unsafe_allow_html=True)
        st.caption("Estimate from historical records. Real yield also depends on weather, soil "
                   "and management.")

# ---------------- WEATHER ----------------
elif page == "⛅ Weather":
    st.title("⛅ Weather & Farming Alerts")
    loc = pick_location("wx")
    if loc:
        try:
            data = fetch_weather(loc["lat"], loc["lon"])
        except Exception:
            st.error("Could not fetch weather. Check your internet connection and try again.")
            st.stop()

        cur, daily = data["current"], data["daily"]
        emoji, desc = wx(cur["weather_code"])
        st.markdown(f"""<div class='wx-main'>
            <div class='city'>📍 {loc['label']}</div>
            <div class='temp'>{emoji} {cur['temperature_2m']:.0f}°C</div>
            <div class='desc'>{desc} · feels like {cur['apparent_temperature']:.0f}°C</div>
            <div class='wx-row'>
              <div>💧 Humidity {cur['relative_humidity_2m']}%</div>
              <div>💨 Wind {cur['wind_speed_10m']} km/h</div>
              <div>🌧️ Rain now {cur['precipitation']} mm</div>
            </div></div>""", unsafe_allow_html=True)

        st.subheader("7-day forecast")
        cols = st.columns(7)
        for i, col in enumerate(cols):
            e, _ = wx(daily["weather_code"][i])
            dname = pd.to_datetime(daily["time"][i]).strftime("%a %d")
            col.markdown(f"<div class='day'><div class='d'>{dname}</div><div class='e'>{e}</div>"
                         f"<div class='t'>{daily['temperature_2m_max'][i]:.0f}° / "
                         f"{daily['temperature_2m_min'][i]:.0f}°</div>"
                         f"<div class='r'>🌧️ {daily['precipitation_sum'][i]:.1f} mm</div></div>",
                         unsafe_allow_html=True)

        st.write("")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Temperature (°C)**")
            st.line_chart(pd.DataFrame({"Max": daily["temperature_2m_max"],
                                        "Min": daily["temperature_2m_min"]},
                                       index=daily["time"]))
        with t2:
            st.markdown("**Rainfall (mm)**")
            st.bar_chart(pd.DataFrame({"Rain": daily["precipitation_sum"]}, index=daily["time"]))

        st.subheader("🚜 Farming alerts")
        tips = []
        if max(daily["temperature_2m_max"]) >= 38:
            tips.append("🔥 **Heat stress likely** (38°C or more). Irrigate in the early morning or evening.")
        if min(daily["temperature_2m_min"]) <= 5:
            tips.append("🥶 **Cold / frost risk.** Protect young plants with mulch or covers at night.")
        if sum(daily["precipitation_sum"]) >= 30:
            tips.append("🌧️ **Heavy rain expected this week.** Check drainage and delay fertiliser application.")
        if max(daily["precipitation_sum"]) < 1:
            tips.append("☀️ **Dry week ahead.** Plan irrigation (see the Irrigation page).")
        if cur["wind_speed_10m"] >= 25:
            tips.append("💨 **Strong wind.** Avoid spraying pesticides today.")
        if cur["relative_humidity_2m"] >= 85:
            tips.append("🍄 **High humidity.** Fungal diseases (blight, mildew) spread faster. Scout your leaves.")
        if not tips:
            tips.append("✅ Conditions look normal this week. No special alerts.")
        for t in tips:
            st.markdown(f"<div class='tip'>{t}</div>", unsafe_allow_html=True)
        st.caption("Data: Open-Meteo.com (free weather API).")

# ---------------- IRRIGATION ----------------
elif page == "💧 Irrigation Advice":
    st.title("💧 Irrigation Advice")
    st.caption("Water need = (reference evapotranspiration × crop coefficient) minus effective rain, "
               "adjusted for irrigation efficiency.")
    loc = pick_location("ir")

    c1, c2 = st.columns(2)
    crop = c1.selectbox("Crop", list(KC.keys()))
    stage = c2.selectbox("Growth stage", STAGES)
    c3, c4 = st.columns(2)
    area = c3.number_input("Field area (hectares)", min_value=0.05, value=1.0, step=0.25)
    method = c4.selectbox("Irrigation method", list(METHODS.keys()))

    if loc:
        try:
            data = fetch_weather(loc["lat"], loc["lon"])
        except Exception:
            st.error("Could not fetch weather. Check your internet connection and try again.")
            st.stop()

        daily = data["daily"]
        kc = KC[crop][STAGES.index(stage)]
        eff = METHODS[method]
        rows = []
        for i in range(7):
            et0 = daily["et0_fao_evapotranspiration"][i] or 0.0
            rain = daily["precipitation_sum"][i] or 0.0
            etc = et0 * kc                          # crop water use, mm/day
            eff_rain = 0.8 * rain                   # effective rainfall
            net = max(0.0, etc - eff_rain)          # net irrigation need, mm
            gross = net / eff                       # after losses
            litres = gross * 10000 * area           # 1 mm over 1 ha = 10,000 L
            rows.append({"Date": daily["time"][i], "Rain (mm)": round(rain, 1),
                         "ET0 (mm)": round(et0, 1), "Crop use (mm)": round(etc, 1),
                         "Irrigate (mm)": round(gross, 1), "Water (litres)": int(round(litres, -1))})
        df = pd.DataFrame(rows)

        today = rows[0]
        if today["Irrigate (mm)"] < 1:
            st.markdown(f"<div class='result blue'><span class='badge'>TODAY</span>"
                        f"<h2>No irrigation needed today</h2><div class='sub'>Rain and low "
                        f"evaporation cover your {crop.lower()}'s water use.</div></div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result ok'><span class='badge'>TODAY</span>"
                        f"<h2>Irrigate about {today['Water (litres)']:,} litres</h2>"
                        f"<div class='sub'>{today['Irrigate (mm)']} mm over {area} ha "
                        f"({method.lower()}) for {crop.lower()}, {stage.split(' (')[0].lower()} stage</div></div>",
                        unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Crop coefficient (Kc)", f"{kc:.2f}")
        m2.metric("Week total water", f"{int(df['Water (litres)'].sum()):,} L")
        m3.metric("Expected rain this week", f"{df['Rain (mm)'].sum():.0f} mm")

        st.markdown("**Daily water need (litres)**")
        st.bar_chart(df.set_index("Date")["Water (litres)"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption("Simplified daily water balance. It does not track soil moisture storage, so "
                   "adjust for your soil type and always check the soil by hand before irrigating. "
                   "Kc values are approximate FAO-56 figures. Weather data: Open-Meteo.com.")

# ---------------- PERFORMANCE ----------------
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    st.header("Leaf disease model")
    st.write("**MobileNetV2** (transfer learning, 2-phase fine-tuning) trained on the "
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

    st.divider()
    st.header("Yield prediction model")
    st.write("Three regression models compared on about 235,000 district-level records. "
             "The target is yield (tonnes per hectare), trained on a log scale. "
             "The best model is a **Random Forest**.")
    yp = ROOT / "docs" / "yield_results.csv"
    if yp.exists():
        st.dataframe(pd.read_csv(yp), use_container_width=True, hide_index=True)
    yi = ROOT / "docs" / "yield_actual_vs_pred.png"
    if yi.exists():
        st.image(str(yi), caption="Actual vs predicted yield (Random Forest, test set)", width=420)
    st.info("The yield test set is a random split, so rows come from the same districts and "
            "years as training. Forecasts for a brand-new season are expected to be less "
            "accurate than the R² shown here.")

# ---------------- ROADMAP ----------------
else:
    st.title("🚀 Roadmap")
    st.success("✅ **Done:** Leaf disease detection (97% test accuracy, 38 classes)")
    st.success("✅ **Done:** Yield prediction (Random Forest, R² 0.94)")
    st.success("✅ **Done:** Weather forecast and farming alerts (Open-Meteo)")
    st.success("✅ **Done:** Irrigation advice (FAO evapotranspiration method)")
    a, b, c = st.columns(3)
    a.markdown("<div class='card soon'><h3>🧪 Real-world testing</h3>Test the disease model on "
               "field photos and report accuracy honestly.</div>", unsafe_allow_html=True)
    b.markdown("<div class='card soon'><h3>🌐 Deployment</h3>Publish the app online so "
               "anyone can open it from a phone.</div>", unsafe_allow_html=True)
    c.markdown("<div class='card soon'><h3>📡 Soil sensors (optional)</h3>ESP32 soil-moisture "
               "sensor feeding live data into irrigation advice.</div>", unsafe_allow_html=True)