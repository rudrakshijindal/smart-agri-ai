HEALTHY = {
    "about": "No disease signs detected. The leaf looks healthy.",
    "action": [
        "Keep your regular watering and nutrient schedule.",
        "Inspect plants weekly, especially after rain or humid weather.",
        "Remove fallen debris and keep good air flow between plants.",
    ],
}

GENERIC = {
    "about": "A disease pattern was detected on this leaf.",
    "action": [
        "Remove and destroy badly affected leaves; do not compost them.",
        "Avoid overhead watering and keep leaves dry.",
        "Consult your local agriculture officer before applying any chemical.",
    ],
}

ADVICE = {
    "apple_scab": ("Fungal disease causing olive-green to brown scabby spots on leaves and fruit.",
        ["Rake and destroy fallen leaves.", "Prune for air circulation.", "Apply a protectant fungicide as per label during wet spring weather."]),
    "black_rot": ("Fungal disease causing dark circular lesions and rotting of fruit.",
        ["Prune out infected shoots and mummified fruit.", "Improve air flow and avoid wet foliage.", "Use a recommended fungicide as per label."]),
    "cedar_apple_rust": ("Fungal rust with bright yellow-orange spots on leaves.",
        ["Remove nearby juniper or cedar hosts if possible.", "Use a rust-active fungicide as per label from pink bud stage."]),
    "powdery_mildew": ("Fungal disease forming white powdery coating on leaves.",
        ["Improve air flow and reduce shade.", "Avoid excess nitrogen.", "Apply sulphur or a recommended fungicide as per label."]),
    "cercospora_leaf_spot gray_leaf_spot": ("Fungal disease causing rectangular grey-tan lesions on maize leaves.",
        ["Rotate crops and till residue.", "Choose resistant hybrids.", "Apply a foliar fungicide if infection is early and severe."]),
    "common_rust_": ("Fungal rust with small reddish-brown pustules on both leaf surfaces.",
        ["Plant resistant varieties.", "Scout early; apply a fungicide only if pustules spread rapidly."]),
    "northern_leaf_blight": ("Fungal disease with long cigar-shaped grey-green lesions.",
        ["Rotate crops and bury residue.", "Use resistant hybrids.", "Apply a foliar fungicide in severe cases."]),
    "esca_(black_measles)": ("Fungal trunk disease causing tiger-stripe leaves and spotted berries.",
        ["Prune out dead wood in dry weather and seal wounds.", "Remove severely affected vines.", "No complete cure; focus on prevention."]),
    "leaf_blight_(isariopsis_leaf_spot)": ("Fungal leaf spot with dark brown angular patches on grape leaves.",
        ["Remove infected leaves.", "Improve canopy air flow.", "Apply a protectant fungicide as per label."]),
    "haunglongbing_(citrus_greening)": ("Bacterial disease spread by psyllid insects causing blotchy yellow leaves and bitter fruit.",
        ["Remove infected trees to protect others.", "Control psyllids with recommended insecticides.", "Plant certified disease-free saplings."]),
    "bacterial_spot": ("Bacterial disease causing small dark water-soaked spots on leaves and fruit.",
        ["Use disease-free seed and rotate crops.", "Avoid working in wet fields.", "Copper-based sprays can reduce spread as per label."]),
    "early_blight": ("Fungal disease with brown spots showing concentric rings, starting on lower leaves.",
        ["Remove lower infected leaves.", "Mulch to stop soil splash and water at the base.", "Apply a protectant fungicide as per label; rotate crops."]),
    "late_blight": ("Fast-spreading water mould causing large dark greasy patches, worst in cool wet weather.",
        ["Remove and destroy infected plants quickly.", "Avoid overhead irrigation.", "Apply a recommended fungicide immediately and repeat as per label."]),
    "leaf_scorch": ("Fungal disease causing purple-brown blotches that make leaf edges look scorched.",
        ["Remove infected leaves.", "Water at the base and space plants well.", "Use resistant varieties and a fungicide if needed."]),
    "leaf_mold": ("Fungal disease causing pale spots on top and olive-brown mould underneath, common in humid conditions.",
        ["Lower humidity and improve ventilation.", "Remove infected leaves.", "Apply a recommended fungicide as per label."]),
    "septoria_leaf_spot": ("Fungal disease with many small round spots with dark borders and grey centres.",
        ["Remove affected lower leaves.", "Mulch and avoid wetting foliage.", "Rotate crops and use a protectant fungicide."]),
    "spider_mites two-spotted_spider_mite": ("Tiny pests causing yellow stippling and fine webbing, worse in hot dry weather.",
        ["Spray plants with water to reduce mites.", "Use neem oil or insecticidal soap.", "Avoid dusty conditions and protect natural predators."]),
    "target_spot": ("Fungal disease with brown spots showing ring patterns.",
        ["Remove infected leaves.", "Improve air flow and avoid overhead watering.", "Apply a recommended fungicide as per label."]),
    "tomato_yellow_leaf_curl_virus": ("Virus spread by whiteflies causing curled, yellow, stunted leaves.",
        ["Control whiteflies with yellow sticky traps and recommended sprays.", "Remove and destroy infected plants.", "Use resistant varieties and insect nets in nurseries."]),
    "tomato_mosaic_virus": ("Virus causing mottled light and dark green leaves and distorted growth.",
        ["Remove infected plants.", "Wash hands and disinfect tools.", "Do not use tobacco near plants; use virus-free seed."]),
}


def get_advice(class_name):
    slug = class_name.split("___")[-1].lower()
    if slug == "healthy":
        return HEALTHY
    if slug in ADVICE:
        about, action = ADVICE[slug]
        return {"about": about, "action": action}
    return GENERIC