import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
names = json.loads((ROOT / "models" / "class_names.json").read_text())
model = tf.keras.models.load_model(ROOT / "models" / "leaf_model.keras")

rows = []
for p in sorted((ROOT / "data" / "real_photos").glob("*")):
    if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue
    true = p.stem.rsplit("_", 1)[0]
    img = Image.open(p).convert("RGB").resize((224, 224))
    probs = model.predict(np.array(img, dtype="float32")[None], verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    rows.append({
        "Image": p.name,
        "True": true,
        "Predicted": names[top[0]],
        "Confidence %": round(float(probs[top[0]]) * 100, 1),
        "Top-1 correct": names[top[0]] == true,
        "Top-3 correct": true in [names[i] for i in top],
    })

if not rows:
    raise SystemExit("No photos found in data\\real_photos. Run make_real_photos.py first.")

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("\nPhotos tested:", len(df))
print("Top-1 accuracy:", round(df["Top-1 correct"].mean() * 100, 1), "%")
print("Top-3 accuracy:", round(df["Top-3 correct"].mean() * 100, 1), "%")
df.to_csv(ROOT / "docs" / "real_world_test.csv", index=False)