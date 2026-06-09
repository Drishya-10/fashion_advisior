import startup
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import pandas as pd
import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from collections import defaultdict
import kagglehub
import requests
from io import BytesIO
import os

app = Flask(__name__)
CORS(app)

# ────────────────────────────────────────────
# LOAD EVERYTHING ONCE AT STARTUP
# ────────────────────────────────────────────

# ── Dataset ──
print("Loading dataset from Kaggle...")
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
STYLES_CSV = None
for root, dirs, files in os.walk(path):
    for f in files:
        if f in ("styles.csv", "style.csv"):
            STYLES_CSV = os.path.join(root, f)
            break

df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")
df["id"] = pd.to_numeric(df["id"], errors="coerce").dropna().astype(int)
df = df.dropna(subset=["id"])
df["id"] = df["id"].astype(int)
print(f"✅ Dataset: {len(df)} products")

# ── Embeddings ──
print("Loading embeddings...")
BASE      = os.path.dirname(os.path.abspath(__file__))
vectors = np.load(os.path.join(BASE, "data", "embedding_vectors.npy"))
ids     = np.load(os.path.join(BASE, "data", "embedding_ids.npy"))
ids       = ids.astype(int)
id_to_idx = {int(pid): idx for idx, pid in enumerate(ids)}
print(f"✅ Embeddings: {vectors.shape}")

# ── CLIP ──
print("Loading CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to(device)
clip_model.eval()
print(f"✅ CLIP ready on {device}")

# ── YOLO ──
print("Loading YOLO...")
yolo_model = YOLO("yolov8n.pt")
print("✅ YOLO ready")

# ── RL preference store ──
preference_store = defaultdict(float)


# ────────────────────────────────────────────
# PAIRING RULES (from outfit_pairer.py)
# ────────────────────────────────────────────

PAIRING_RULES = {
    "Topwear":    ["Bottomwear", "Shoes", "Sandal", "Flip Flops", "Watches",
                   "Belts", "Bags", "Jewellery", "Eyewear", "Socks", "Ties",
                   "Scarves", "Mufflers", "Headwear"],
    "Bottomwear": ["Topwear", "Shoes", "Sandal", "Flip Flops", "Belts",
                   "Socks", "Watches", "Bags", "Eyewear"],
    "Dress":      ["Shoes", "Sandal", "Flip Flops", "Bags", "Jewellery",
                   "Watches", "Eyewear", "Scarves", "Headwear"],
    "Saree":      ["Shoes", "Sandal", "Bags", "Jewellery", "Watches",
                   "Eyewear", "Scarves"],
    "Shoes":      ["Topwear", "Bottomwear", "Dress", "Socks",
                   "Watches", "Bags", "Eyewear"],
    "Sandal":     ["Topwear", "Bottomwear", "Dress", "Saree",
                   "Bags", "Jewellery", "Watches"],
    "Flip Flops": ["Topwear", "Bottomwear", "Bags", "Watches", "Eyewear"],
    "Bags":       ["Topwear", "Bottomwear", "Dress", "Shoes",
                   "Sandal", "Watches", "Eyewear"],
    "Watches":    ["Topwear", "Bottomwear", "Dress", "Shoes", "Bags"],
    "Jewellery":  ["Topwear", "Dress", "Saree", "Bags", "Eyewear"],
    "Belts":      ["Topwear", "Bottomwear", "Shoes"],
    "Socks":      ["Topwear", "Bottomwear", "Shoes"],
    "Ties":       ["Topwear", "Bottomwear", "Shoes", "Watches"],
    "Headwear":   ["Topwear", "Bottomwear", "Shoes", "Eyewear"],
    "Scarves":    ["Topwear", "Bottomwear", "Bags"],
    "Mufflers":   ["Topwear", "Bottomwear", "Shoes"],
    "Eyewear":    ["Topwear", "Bottomwear", "Dress", "Shoes", "Bags", "Watches"],
    "Innerwear":  ["Topwear", "Bottomwear", "Socks"],
    "Apparel Set":["Shoes", "Sandal", "Bags", "Watches", "Eyewear"],
}

NEUTRAL_COLORS = {
    "Black", "White", "Grey", "Navy Blue", "Beige", "Cream",
    "Off White", "Charcoal", "Khaki", "Brown", "Tan", "Silver", "Steel"
}

COLOR_HARMONY = {
    "Red":      ["Black", "White", "Navy Blue", "Grey", "Beige"],
    "Blue":     ["White", "Grey", "Black", "Beige", "Brown"],
    "Green":    ["White", "Black", "Beige", "Brown", "Khaki"],
    "Yellow":   ["White", "Black", "Grey", "Navy Blue"],
    "Pink":     ["White", "Black", "Grey", "Navy Blue", "Beige"],
    "Orange":   ["White", "Black", "Navy Blue", "Brown"],
    "Purple":   ["White", "Black", "Grey", "Beige"],
    "Maroon":   ["White", "Black", "Beige", "Khaki"],
    "Mustard":  ["White", "Black", "Brown", "Beige", "Navy Blue"],
    "Rust":     ["White", "Black", "Brown", "Beige", "Khaki"],
    "Turquoise":["White", "Black", "Navy Blue", "Beige"],
    "Lavender": ["White", "Grey", "Black", "Beige"],
    "Olive":    ["White", "Black", "Brown", "Beige", "Khaki"],
}

# YOLO labels to ignore
IGNORE_LABELS = {
    "person", "car", "truck", "bus", "bicycle", "motorcycle",
    "chair", "couch", "dining table", "bed", "laptop",
    "cell phone", "book", "bottle", "cat", "dog", "bird"
}

# COCO label → dataset subCategory
COCO_TO_SUBCAT = {
    "tie":      "Ties",
    "backpack": "Bags",
    "handbag":  "Bags",
    "suitcase": "Bags",
}


# ────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────

def colors_match(c1: str, c2: str) -> bool:
    c1, c2 = str(c1).strip(), str(c2).strip()
    if c1 in NEUTRAL_COLORS or c2 in NEUTRAL_COLORS:
        return True
    if c1 == c2:
        return True
    return c2 in COLOR_HARMONY.get(c1, [])


def _embed_image(img: Image.Image) -> np.ndarray:
    """Generate CLIP embedding for a PIL image."""
    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def detect_and_embed(img: Image.Image) -> dict:
    """
    From yolo.py — YOLO detects item, CLIP embeds the crop.
    Returns best detection with embedding.
    """
    img_rgb   = img.convert("RGB")
    results   = yolo_model(img_rgb, verbose=False)
    best      = None
    best_conf = 0.0

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = result.names[cls]

            if label in IGNORE_LABELS or conf < 0.25:
                continue

            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = img_rgb.size
                pad  = 20
                crop = img_rgb.crop((
                    max(0, x1-pad), max(0, y1-pad),
                    min(w, x2+pad), min(h, y2+pad)
                ))
                best = {
                    "label":      label,
                    "subcat":     COCO_TO_SUBCAT.get(label, "Topwear"),
                    "confidence": round(conf, 3),
                    "crop":       crop,
                }

    # Fallback — embed full image
    if best is None:
        best = {
            "label":      "clothing",
            "subcat":     "Topwear",
            "confidence": 0.0,
            "crop":       img_rgb,
        }

    best["embedding"] = _embed_image(best["crop"])
    return best


def find_similar_in_dataset(
    embedding:  np.ndarray,
    subcategory: str,
    gender:     str = None,
    usage:      str = None,
    top_n:      int = 3,
) -> list:
    """
    From yolo.py — find most visually similar items
    in a given subcategory using CLIP + RL scores.
    """
    pool = df[df["subCategory"] == subcategory].copy()
    if gender:
        pool = pool[pool["gender"].isin([gender, "Unisex"])]
    if usage:
        pool = pool[pool["usage"] == usage]
    pool = pool.drop_duplicates(subset="productDisplayName")

    if pool.empty:
        pool = df[df["subCategory"] == subcategory].drop_duplicates(
            subset="productDisplayName"
        )

    pool_ids = [int(pid) for pid in pool["id"].tolist() if int(pid) in id_to_idx]
    if not pool_ids:
        return []

    pool_vecs   = vectors[[id_to_idx[pid] for pid in pool_ids]]
    clip_scores = cosine_similarity(embedding.reshape(1, -1), pool_vecs)[0]

    # Apply RL preference scores
    final = np.array([
        clip_scores[i] + preference_store.get(pid, 0.0)
        for i, pid in enumerate(pool_ids)
    ])

    top_idx = np.argsort(final)[::-1][:top_n]
    results = []
    for i in top_idx:
        pid  = pool_ids[i]
        rows = df[df["id"] == pid]
        if rows.empty:
            continue
        row = rows.iloc[0]
        results.append({
            "id":    int(pid),
            "name":  str(row["productDisplayName"]),
            "color": str(row["baseColour"]),
            "subcat": str(row["subCategory"]),
            "score": round(float(final[i]), 4),
        })
    return results


def get_outfit_clip(product_id: int, top_n: int = 3) -> dict:
    """
    From clip_embedding.py — rule engine + CLIP combined.
    Given a matched product ID, build full outfit suggestions.
    """
    item_rows = df[df["id"] == product_id]
    if item_rows.empty:
        return {"error": f"Product {product_id} not found"}

    item    = item_rows.iloc[0]
    sub_cat = str(item["subCategory"])
    gender  = str(item["gender"])
    season  = str(item["season"]) if pd.notna(item["season"]) else None
    usage   = str(item["usage"])
    color   = str(item["baseColour"])

    compatible_cats = PAIRING_RULES.get(sub_cat, [])
    if not compatible_cats:
        return {"error": f"No pairing rules for '{sub_cat}'"}

    outfit = {
        "selected_item": {
            "id":       int(product_id),
            "name":     str(item["productDisplayName"]),
            "color":    color,
            "subcat":   sub_cat,
        },
        "suggestions": {}
    }

    for cat in compatible_cats:
        # Rule-based pre-filter (from outfit_pairer.py)
        pool = df[df["subCategory"] == cat].copy()
        pool = pool[pool["gender"].isin([gender, "Unisex"])]
        if season:
            pool = pool[
                pool["season"].isin([season, "All Season"]) | pool["season"].isna()
            ]
        pool = pool[pool["usage"] == usage]
        pool = pool[pool["baseColour"].apply(lambda c: colors_match(color, str(c)))]
        pool = pool.drop_duplicates(subset="productDisplayName")

        # Relax filters if too few
        if len(pool) < top_n:
            pool = df[df["subCategory"] == cat].copy()
            pool = pool[pool["gender"].isin([gender, "Unisex"])]
            pool = pool[pool["baseColour"].apply(lambda c: colors_match(color, str(c)))]
            pool = pool.drop_duplicates(subset="productDisplayName")

        if pool.empty or product_id not in id_to_idx:
            continue

        # CLIP similarity ranking
        query_vec   = vectors[id_to_idx[product_id]].reshape(1, -1)
        pool_ids    = [int(pid) for pid in pool["id"].tolist() if int(pid) in id_to_idx]
        if not pool_ids:
            continue

        pool_vecs   = vectors[[id_to_idx[pid] for pid in pool_ids]]
        clip_scores = cosine_similarity(query_vec, pool_vecs)[0]
        final       = np.array([
            clip_scores[i] + preference_store.get(pid, 0.0)
            for i, pid in enumerate(pool_ids)
        ])
        top_idx = np.argsort(final)[::-1][:top_n]

        suggestions = []
        for i in top_idx:
            pid  = pool_ids[i]
            rows = df[df["id"] == pid]
            if rows.empty:
                continue
            row = rows.iloc[0]
            suggestions.append({
                "id":    int(pid),
                "name":  str(row["productDisplayName"]),
                "color": str(row["baseColour"]),
                "score": round(float(final[i]), 4),
            })

        if suggestions:
            outfit["suggestions"][cat] = suggestions

    return outfit


# ────────────────────────────────────────────
# FLASK ROUTES
# ────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "products": len(df), "embeddings": len(ids)})


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Main endpoint — receives uploaded image,
    runs YOLO + CLIP + outfit pairing,
    returns complete outfit suggestions.
    """
    try:
        # Get image from request
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file   = request.files["file"]
        gender = request.form.get("gender", "Unisex")
        usage  = request.form.get("usage",  "Casual")

        img = Image.open(file.stream).convert("RGB")

        # Step 1 — YOLO detect + CLIP embed (from yolo.py)
        detection = detect_and_embed(img)
        print(f"Detected: {detection['label']} → {detection['subcat']} "
              f"(conf: {detection['confidence']})")

        # Step 2 — Find closest match in dataset
        matches = find_similar_in_dataset(
            detection["embedding"],
            subcategory=detection["subcat"],
            gender=gender,
            usage=usage,
            top_n=1,
        )

        # Fallback — try without filters
        if not matches:
            matches = find_similar_in_dataset(
                detection["embedding"],
                subcategory="Topwear",
                top_n=1,
            )

        if not matches:
            return jsonify({"error": "No matching items found"}), 404

        matched_item = matches[0]
        matched_id   = matched_item["id"]
        print(f"Matched: {matched_item['name']} (score: {matched_item['score']})")

        # Step 3 — Build full outfit using clip+pairing engine
        outfit_result = get_outfit_clip(matched_id, top_n=3)

        if "error" in outfit_result:
            # Fallback — use simple similarity for all categories
            suggestions = {}
            for cat in PAIRING_RULES.get(detection["subcat"], []):
                items = find_similar_in_dataset(
                    detection["embedding"], cat,
                    gender=gender, usage=usage, top_n=3
                )
                if items:
                    suggestions[cat] = items

            return jsonify({
                "detected":     detection["label"],
                "confidence":   detection["confidence"],
                "matched_item": matched_item,
                "suggestions":  suggestions,
            })

        return jsonify({
            "detected":     detection["label"],
            "confidence":   detection["confidence"],
            "matched_item": matched_item,
            "suggestions":  outfit_result["suggestions"],
        })

    except Exception as e:
        print(f"Error in /recommend: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    RL feedback — like/dislike updates preference scores
    which influence future recommendations.
    """
    try:
        data       = request.json
        product_id = int(data["product_id"])
        signal     = data["signal"]  # 'like' or 'dislike'

        weight = 0.3 if signal == "like" else -0.3
        preference_store[product_id] += weight

        # Propagate to visually similar items
        if product_id in id_to_idx:
            query  = vectors[id_to_idx[product_id]].reshape(1, -1)
            scores = cosine_similarity(query, vectors)[0]
            top_i  = np.argsort(scores)[::-1][1:6]
            prop   = 0.1 if signal == "like" else -0.1
            for i in top_i:
                preference_store[int(ids[i])] += prop

        return jsonify({
            "status":    "ok",
            "product_id": product_id,
            "signal":    signal,
            "new_score": round(preference_store[product_id], 3),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────
# CHATBOT (from chatbot.py)
# ────────────────────────────────────────────

from groq import Groq
import json

groq_client          = Groq(api_key=os.getenv("GROQ_API_KEY"))
conversation_history = []

LIKE_WEIGHT    =  0.3
DISLIKE_WEIGHT = -0.3


def llm(messages: list, temperature: float = 0.7) -> str:
    """Single Groq LLM call."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=temperature,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def extract_intent(message: str) -> dict:
    """Parse natural language into structured fashion intent."""
    system = """You are a fashion intent parser.
Return ONLY valid JSON — no markdown, no explanation:
{
  "gender": "Men"|"Women"|"Unisex"|null,
  "usage":  "Casual"|"Formal"|"Sports"|"Ethnic"|"Party"|null,
  "season": "Summer"|"Winter"|"Fall"|"Spring"|null,
  "occasion": string|null,
  "base_item": string|null,
  "refine": string|null
}"""
    raw = llm([
        {"role": "system", "content": system},
        {"role": "user",   "content": message},
    ], temperature=0)
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except Exception:
        return {}


def _propagate_feedback(product_id: int, weight: float, top_k: int = 5):
    """Spread preference signal to visually similar items."""
    if product_id not in id_to_idx:
        return
    query  = vectors[id_to_idx[product_id]].reshape(1, -1)
    scores = cosine_similarity(query, vectors)[0]
    top_i  = np.argsort(scores)[::-1][1:top_k + 1]
    for i in top_i:
        preference_store[int(ids[i])] += weight


def suggest_outfit_from_text(intent: dict, base_product_id: int = None, top_n: int = 3) -> dict:
    """Build outfit from text intent — used by chatbot."""
    gender = intent.get("gender")
    usage  = intent.get("usage")
    season = intent.get("season")

    if base_product_id is None:
        pool = df.copy()
        if gender: pool = pool[pool["gender"].isin([gender, "Unisex"])]
        if usage:  pool = pool[pool["usage"] == usage]
        if season: pool = pool[pool["season"] == season]
        pool = pool[pool["subCategory"].isin(PAIRING_RULES.keys())]
        if pool.empty:
            pool = df[df["subCategory"].isin(["Topwear", "Bottomwear", "Dress"])]
        if pool.empty:
            return {"error": "No matching item found"}
        base_row        = pool.sample(1).iloc[0]
        base_product_id = int(base_row["id"])
    else:
        rows = df[df["id"] == base_product_id]
        if rows.empty:
            return {"error": "Product not found"}
        base_row = rows.iloc[0]

    sub_cat = str(base_row["subCategory"])
    color   = str(base_row["baseColour"])
    gender  = gender or str(base_row["gender"])
    usage   = usage  or str(base_row["usage"])
    season  = season or (str(base_row["season"]) if pd.notna(base_row["season"]) else None)

    outfit = {
        "base_item": {
            "id":       int(base_product_id),
            "name":     str(base_row["productDisplayName"]),
            "color":    color,
            "category": sub_cat,
        },
        "suggestions": {}
    }

    for cat in PAIRING_RULES.get(sub_cat, []):
        pool = df[df["subCategory"] == cat].copy()
        pool = pool[pool["gender"].isin([gender, "Unisex"])]
        if usage:  pool = pool[pool["usage"] == usage]
        if season: pool = pool[pool["season"].isin([season, "All Season"]) | pool["season"].isna()]
        pool = pool[pool["baseColour"].apply(lambda c: colors_match(color, str(c)))]
        pool = pool.drop_duplicates(subset="productDisplayName")

        if len(pool) < top_n:
            pool = df[df["subCategory"] == cat].copy()
            pool = pool[pool["gender"].isin([gender, "Unisex"])]
            pool = pool[pool["baseColour"].apply(lambda c: colors_match(color, str(c)))]
            pool = pool.drop_duplicates(subset="productDisplayName")

        if pool.empty or base_product_id not in id_to_idx:
            continue

        query_vec   = vectors[id_to_idx[base_product_id]].reshape(1, -1)
        pool_ids    = [int(pid) for pid in pool["id"].tolist() if int(pid) in id_to_idx]
        if not pool_ids:
            continue

        pool_vecs   = vectors[[id_to_idx[pid] for pid in pool_ids]]
        clip_scores = cosine_similarity(query_vec, pool_vecs)[0]
        final       = np.array([
            clip_scores[i] + preference_store.get(pid, 0.0)
            for i, pid in enumerate(pool_ids)
        ])
        top_idx = np.argsort(final)[::-1][:top_n]

        outfit["suggestions"][cat] = [
            {
                "id":    int(pool_ids[i]),
                "name":  str(df[df["id"] == pool_ids[i]].iloc[0]["productDisplayName"]),
                "color": str(df[df["id"] == pool_ids[i]].iloc[0]["baseColour"]),
                "score": round(float(final[i]), 4),
            }
            for i in top_idx
            if not df[df["id"] == pool_ids[i]].empty
        ]

    return outfit


# ── Chatbot routes ──

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Receive message → extract intent → suggest outfit → Zara responds.
    """
    global conversation_history
    try:
        data            = request.json
        message         = data.get("message", "")
        base_product_id = data.get("base_product_id", None)

        if not message:
            return jsonify({"error": "No message provided"}), 400

        intent = extract_intent(message)
        outfit = suggest_outfit_from_text(intent, base_product_id)

        if "error" in outfit:
            return jsonify({
                "reply": f"Sorry — {outfit['error']}. Could you describe what you'd like to wear?",
                "outfit": None
            })

        # Build outfit summary for LLM
        base    = outfit["base_item"]
        summary = f"Base: {base['name']} ({base['color']})\n"
        for cat, items in outfit["suggestions"].items():
            summary += f"\n{cat}:\n"
            for item in items:
                summary += f"  - {item['name']} ({item['color']})\n"

        system = """You are Zara, a warm confident AI fashion stylist.
Respond naturally like a personal stylist — concise, mention product names,
never mention scores. End with one short question to refine further."""

        conversation_history.append({"role": "user", "content": message})
        messages = (
            [{"role": "system", "content": system}]
            + conversation_history[-6:]
            + [{"role": "user", "content": f"Present this outfit:\n{summary}"}]
        )
        reply = llm(messages, temperature=0.75)
        conversation_history.append({"role": "assistant", "content": reply})

        return jsonify({"reply": reply, "outfit": outfit})

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback_endpoint():
    """
    Like/dislike feedback — updates RL preference store.
    """
    try:
        data       = request.json
        product_id = int(data["product_id"])
        signal     = data["signal"]

        if signal not in ("like", "dislike"):
            return jsonify({"error": "signal must be like or dislike"}), 400

        weight = LIKE_WEIGHT if signal == "like" else DISLIKE_WEIGHT
        preference_store[product_id] += weight
        _propagate_feedback(
            product_id,
            0.1 if signal == "like" else -0.1
        )

        return jsonify({
            "status":     "ok",
            "product_id": product_id,
            "signal":     signal,
            "new_score":  round(preference_store[product_id], 3),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/clear", methods=["DELETE"])
def clear_history():
    """Reset conversation with Zara."""
    conversation_history.clear()
    return jsonify({"status": "conversation cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
