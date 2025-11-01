import os, io, re, logging, numpy as np, joblib, tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pdfminer.high_level import extract_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")
CORS(app)

# Globals
model = None
scaler = None
label_encoder = None
knowledge_graph = None
SYSTEM_READY = False
EXPECTED = 561

def ensure_model_exists():
    if os.path.exists("military_screening_cnn.h5"):
        return True
    try:
        import py7zr
        if os.path.exists("military_screening_cnn.7z"):
            log.info("🔄 Extracting model from 7z...")
            with py7zr.SevenZipFile("military_screening_cnn.7z", "r") as z:
                z.extractall()
            log.info("✅ Model extracted")
            return os.path.exists("military_screening_cnn.h5")
        else:
            log.error("❌ 7z archive not found")
            return False
    except Exception as e:
        log.error(f"❌ Extraction failed: {e}")
        return False

def default_kg():
    # simple dict (no pickled class needed)
    return {
        "role_rules": {
            "LOW": ["Infantry","Special Forces","Combat Engineer"],
            "MODERATE": ["Military Police","Logistics","Signals","Administration"],
            "HIGH": ["Medical Evaluation Required"]
        }
    }

def load_all():
    global model, scaler, label_encoder, knowledge_graph, SYSTEM_READY
    log.info("🚀 Loading components...")
    if not ensure_model_exists():
        log.warning("Model not available yet")
    if os.path.exists("military_screening_cnn.h5"):
        model = tf.keras.models.load_model("military_screening_cnn.h5", compile=False)
        try: model.compile()
        except Exception: pass
        log.info("✅ Model loaded")

    try:
        scaler = joblib.load("scaler.pkl"); log.info("✅ Scaler loaded")
    except Exception as e:
        log.warning(f"⚠️ Scaler load failed: {e}"); scaler = None

    try:
        label_encoder = joblib.load("label_encoder.pkl"); log.info("✅ Label encoder loaded")
    except Exception as e:
        log.warning(f"⚠️ Label encoder load failed: {e}"); label_encoder = None

    try:
        kg_obj = joblib.load("military_knowledge_graph.pkl")
        # if it provides a method, keep; else fall back to dict
        knowledge_graph = kg_obj if isinstance(kg_obj, dict) else kg_obj
        log.info("✅ Knowledge graph loaded")
    except Exception as e:
        log.warning(f"⚠️ KG load failed: {e}; using default")
        knowledge_graph = default_kg()

    # Warmup to prevent first-request slowness
    if model is not None:
        try:
            dummy = np.zeros((1, EXPECTED), dtype=np.float32)
            if scaler is not None:
                dummy = scaler.transform(dummy)
            dummy = dummy.reshape((1, EXPECTED, 1))
            _ = model.predict(dummy, verbose=0)
            log.info("🔥 Warmup inference complete")
        except Exception as e:
            log.warning(f"Warmup skipped: {e}")

    # App is ready if the model is ready (scaler/LE are optional)
    SYSTEM_READY = model is not None
    if SYSTEM_READY:
        log.info("🎯 System ready")
    else:
        log.error("❌ System not ready (model missing)")

def _roles_from_kg(confidence, biomarkers):
    """Try object-style KG; else dict fallback."""
    try:
        # object style: has recommend_roles(biomarkers) or get_recommendations(confidence)
        if hasattr(knowledge_graph, "recommend_roles"):
            res = knowledge_graph.recommend_roles(biomarkers)
            return res.get("recommended_roles", []), res.get("detected_risks", [])
        if hasattr(knowledge_graph, "get_recommendations"):
            return list(knowledge_graph.get_recommendations(confidence)), []
    except Exception as e:
        log.warning(f"KG error: {e}")
    # dict fallback
    risk = "LOW" if confidence>0.8 else "MODERATE" if confidence>0.6 else "HIGH"
    return knowledge_graph.get("role_rules", {}).get(risk, ["General Service"]), []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy" if SYSTEM_READY else "initializing",
        "system_ready": SYSTEM_READY,
        "components": {
            "model": bool(model is not None),
            "scaler": bool(scaler is not None),
            "label_encoder": bool(label_encoder is not None),
            "knowledge_graph": bool(knowledge_graph is not None)
        }
    })

def _predict_core(vec):
    arr = np.array(vec, dtype=np.float32).reshape(1, -1)
    if scaler is not None:
        arr = scaler.transform(arr)
    X = arr.reshape((1, EXPECTED, 1))
    probs = model.predict(X, verbose=0)
    if probs.ndim == 2:
        conf = float(np.max(probs[0]))
        idx = int(np.argmax(probs[0]))
    else:
        conf = float(probs.squeeze()); idx = int(conf >= 0.5)
    try:
        activity = str(label_encoder.inverse_transform([idx])[0]) if label_encoder is not None else str(idx)
    except Exception:
        activity = str(idx)

    biomarkers = {
        "movement_quality": conf,
        "fatigue_index": 0.05 if conf>0.8 else 0.15 if conf>0.6 else 0.25,
        "movement_smoothness": conf*0.9 + 0.1
    }
    decision, risk, reason = (
        ("PASS","LOW","Excellent movement quality and physical performance") if conf>0.8 else
        ("CONDITIONAL PASS","MODERATE","Adequate performance with some areas for improvement") if conf>0.6 else
        ("FAIL","HIGH","Movement analysis indicates physical limitations")
    )
    roles, risks = _roles_from_kg(conf, biomarkers)
    return {
        "activity": activity,
        "confidence": conf,
        "decision": decision,
        "reason": reason,
        "risk_level": risk,
        "recommended_roles": roles,
        "detected_risks": risks,
        "performance_score": round(conf*100,1)
    }

@app.route("/predict", methods=["POST","OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200
    if not SYSTEM_READY:
        return jsonify({"success":False,"error":"System initializing; retry shortly."}), 503
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"success":False,"error":"Invalid JSON body"}), 400
    if not payload or "sensor_data" not in payload:
        return jsonify({"success":False,"error":"Missing 'sensor_data'"}), 400

    data = payload["sensor_data"]
    if not isinstance(data, list) or len(data)!=EXPECTED:
        return jsonify({"success":False,"error":f"'sensor_data' must be a list of {EXPECTED} numbers"}), 400
    try:
        arr = np.array(data, dtype=np.float32); 
        if np.isnan(arr).any() or np.isinf(arr).any():
            return jsonify({"success":False,"error":"sensor_data contains NaN/Inf"}), 400
        pred = _predict_core(arr.tolist())
        return jsonify({"success":True,"prediction":pred})
    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"success":False,"error":f"Server error during prediction: {e}"}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if not SYSTEM_READY:
        return jsonify({"success":False,"error":"System initializing; retry shortly."}), 503
    if "file" not in request.files:
        return jsonify({"success":False,"error":"Missing file"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"success":False,"error":"Only PDF files are accepted"}), 400
    try:
        content = f.read()
        text = extract_text(io.BytesIO(content))
        # Try JSON block
        m = re.search(r"SENSOR_DATA_JSON\\s*:\\s*(\\{.*?\\})", text, flags=re.DOTALL)
        vec = None
        if m:
            import json as _json
            try:
                obj=_json.loads(m.group(1)); cand=obj.get("sensor_data")
                if isinstance(cand,list) and len(cand)==EXPECTED:
                    vec=[float(v) for v in cand]
            except Exception: pass
        if vec is None:
            # Try CSV block
            m = re.search(r"SENSOR_DATA_CSV\\s*:\\s*([\\d\\s,\\.\\-eE]+)", text)
            if m:
                try:
                    cand=[float(v) for v in m.group(1).replace("\\n"," ").split(",") if v.strip()!=""]
                    if len(cand)==EXPECTED: vec=cand
                except Exception: pass
        if vec is None:
            return jsonify({"success":False,"error":"Could not find SENSOR_DATA_JSON or SENSOR_DATA_CSV in PDF"}), 400
        pred = _predict_core(vec)
        return jsonify({"success":True,"prediction":pred})
    except Exception as e:
        log.exception("Upload error")
        return jsonify({"success":False,"error":f"Failed to process PDF: {e}"}), 500

# Load everything at import time
load_all()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
