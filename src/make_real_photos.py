import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "plantdoc"
OUT = ROOT / "data" / "real_photos"
PER_CLASS = 3

names = json.loads((ROOT / "models" / "class_names.json").read_text())

# (keywords that must all appear in the PlantDoc folder name, target class). Order matters.
RULES = [
    (["tomato", "late", "blight"], "Tomato___Late_blight"),
    (["tomato", "early", "blight"], "Tomato___Early_blight"),
    (["tomato", "septoria"], "Tomato___Septoria_leaf_spot"),
    (["tomato", "bacterial"], "Tomato___Bacterial_spot"),
    (["tomato", "mold"], "Tomato___Leaf_Mold"),
    (["tomato", "mosaic"], "Tomato___Tomato_mosaic_virus"),
    (["tomato", "yellow"], "Tomato___Tomato_Yellow_Leaf_Curl_Virus"),
    (["tomato", "mite"], "Tomato___Spider_mites Two-spotted_spider_mite"),
    (["potato", "early", "blight"], "Potato___Early_blight"),
    (["potato", "late", "blight"], "Potato___Late_blight"),
    (["apple", "scab"], "Apple___Apple_scab"),
    (["apple", "rust"], "Apple___Cedar_apple_rust"),
    (["apple", "black", "rot"], "Apple___Black_rot"),
    (["corn", "gray"], "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"),
    (["corn", "rust"], "Corn_(maize)___Common_rust_"),
    (["corn", "blight"], "Corn_(maize)___Northern_Leaf_Blight"),
    (["grape", "black", "rot"], "Grape___Black_rot"),
    (["squash", "powdery"], "Squash___Powdery_mildew"),
    (["pepper", "spot"], "Pepper,_bell___Bacterial_spot"),
]
HEALTHY = {
    "tomato leaf": "Tomato___healthy",
    "apple leaf": "Apple___healthy",
    "grape leaf": "Grape___healthy",
    "bell pepper leaf": "Pepper,_bell___healthy",
    "peach leaf": "Peach___healthy",
    "strawberry leaf": "Strawberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy",
    "blueberry leaf": "Blueberry___healthy",
    "raspberry leaf": "Raspberry___healthy",
    "soyabean leaf": "Soybean___healthy",
    "soybean leaf": "Soybean___healthy",
    "potato leaf": "Potato___healthy",
}


def norm(s):
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()


def target_for(folder):
    n = norm(folder)
    if n in HEALTHY:
        return HEALTHY[n]
    for kws, cls in RULES:
        if all(k in n for k in kws):
            return cls
    return None


if not SRC.exists():
    raise SystemExit(f"Folder not found: {SRC}")

OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*"):
    old.unlink()

count, per = 0, {}
for d in sorted(p for p in SRC.rglob("*") if p.is_dir()):
    cls = target_for(d.name)
    if not cls:
        continue
    if cls not in names:
        print("Skipping (not a model class):", cls)
        continue
    imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for f in imgs:
        if per.get(cls, 0) >= PER_CLASS:
            break
        per[cls] = per.get(cls, 0) + 1
        shutil.copy(f, OUT / f"{cls}_{per[cls]}{f.suffix.lower()}")
        count += 1

print(f"Copied {count} photos for {len(per)} classes into {OUT}")
for cls, n in sorted(per.items()):
    print(f"  {n} x {cls}")