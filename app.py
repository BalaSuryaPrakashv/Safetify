import os
import sys
import json
from dotenv import load_dotenv
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from preprocess import Preprocessor

MODEL_PATH     = "models/bilstm_model.h5"
TOKENIZER_PATH = "tokenizer/tokenizer.pkl"

app = Flask(__name__)
CORS(app)

model        = None
preprocessor = None


# ── Load artifacts ────────────────────────────────────────────────────────────
def load_artifacts():
    global model, preprocessor
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run: python src/train.py"
        )
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found at '{TOKENIZER_PATH}'. Run: python src/train.py"
        )
    model        = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = Preprocessor.load(TOKENIZER_PATH)
    print("Model and preprocessor loaded successfully.")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


DANGER_KEYWORDS = [
    "help", "save me", "danger", "police", "emergency",
    "attack", "attacked", "scared", "afraid", "trapped",
    "rescue", "hurt", "killing", "rape", "molest",
    "run", "escape", "bleeding", "unconscious", "fire"
]

@app.route("/predict", methods=["POST"])
def predict():
    data      = request.get_json(force=True)
    text      = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Keyword override — single critical words trigger DANGER immediately
    text_lower = text.lower()
    keyword_hit = any(kw in text_lower for kw in DANGER_KEYWORDS)

    threshold = getattr(preprocessor, "threshold", 0.5)
    sequence  = preprocessor.transform([text])
    prob      = float(model.predict(sequence, verbose=0)[0][0])

    danger     = keyword_hit or (prob >= threshold)
    label      = "DANGER" if danger else "NORMAL"
    confidence = prob if danger else 1.0 - prob

    return jsonify({
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "raw_score":  round(prob, 4),
        "threshold":  round(threshold, 2),
        "text":       text,
        "keyword_hit": keyword_hit,
    })


@app.route("/send-sos", methods=["POST"])
def send_sos():
    data     = request.get_json(force=True)
    lat      = data.get("latitude")
    lng      = data.get("longitude")
    accuracy = data.get("accuracy", 0)
    phrase   = data.get("phrase", "")
    contacts = data.get("contacts", [])   # list of {name, phone}

    if not contacts:
        # Fall back to env contacts
        env_phones = os.getenv("EMERGENCY_CONTACTS", "")
        env_name   = os.getenv("EMERGENCY_CONTACT_NAME", "Contact")
        contacts   = [
            {"name": env_name, "phone": p.strip()}
            for p in env_phones.split(",") if p.strip()
        ]

    if not contacts:
        return jsonify({"success": False, "error": "No emergency contacts configured"}), 400

    # Build Google Maps link
    if lat and lng:
        maps_url = f"https://maps.google.com/?q={lat},{lng}"
        location_text = f"Location: {maps_url} (+/-{round(accuracy)}m)"
    else:
        maps_url      = None
        location_text = "Location unavailable"

    message = (
        f"SAFETIFY EMERGENCY ALERT\n"
        f"Distress phrase detected: \"{phrase}\"\n"
        f"{location_text}\n"
        f"Please respond immediately or call 100."
    )

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token  = os.getenv("TWILIO_AUTH_TOKEN")

    results = []

    # Dev mode — no credentials
    if not (account_sid and auth_token):
        print("[DEV MODE] Simulating SMS — no Twilio credentials found.")
        results.append({"to": "+916382268580", "name": "Emergency Contact", "success": True, "sid": "dev-mode"})
        return jsonify({
            "success":  True,
            "dev_mode": True,
            "message":  "SOS simulated (add Twilio credentials in .env to send real SMS)",
            "sms_text": message,
            "maps_url": maps_url,
            "details":  results,
        })

    # Twilio SMS send
    from twilio.rest import Client
    client = Client(account_sid, auth_token)

    SMS_FROM ="+16562680832"
    SMS_TO   ="+916382268580"

    try:
        msg = client.messages.create(
            body=message,
            from_=SMS_FROM,
            to=SMS_TO,
        )
        results.append({"to": SMS_TO, "name": "Emergency Contact", "success": True, "sid": msg.sid})
        print(f"SMS sent to {SMS_TO} — SID: {msg.sid}")
    except Exception as e:
        results.append({"to": SMS_TO, "name": "Emergency Contact", "success": False, "error": str(e)})
        print(f"Failed to send SMS to {SMS_TO}: {e}")

    any_ok = any(r["success"] for r in results)
    return jsonify({
        "success":  any_ok,
        "dev_mode": False,
        "message":  "SMS SOS sent" if any_ok else "SMS failed",
        "sms_text": message,
        "maps_url": maps_url,
        "details":  results,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

load_artifacts()

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
   
    app.run(debug=False, host="0.0.0.0", port=5000)
