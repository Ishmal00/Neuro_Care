import re
from pathlib import Path

import chromadb
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEMO_DATA_PATH = BASE_DIR / "data" / "seizeit2_features.csv"
DEMO_MODEL_PATH = MODEL_DIR / "xgb_seizeit2.pkl"
PAPERS_DIR = BASE_DIR / "data" / "papers"
CHROMA_DIR = BASE_DIR / "rag_index" / "chroma"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PATIENT_ID = "sub-002"

API_FEATURE_ORDER = [
    "HR_mean",
    "HRV",
    "EEG_alpha",
    "EEG_beta",
    "EEG_gamma",
    "EEG_delta",
    "EEG_theta",
    "Temperature",
    "Tremor_index",
    "Respiratory_rate",
    "ACC_energy",
]

MODEL_PATHS = {
    patient_id: MODEL_DIR / f"xgb_seizeit2_{patient_id}.pkl"
    for patient_id in ("sub-001", "sub-002", "sub-003")
}

app = Flask(
    __name__,
    static_folder="frontend",
    template_folder="frontend",
)
CORS(app, resources={r"/*": {"origins": "*"}})


def load_saved_model(path):
    saved = joblib.load(path)
    if isinstance(saved, dict) and "model" in saved:
        return saved["model"], saved.get("feature_columns")
    return saved, None


def load_models():
    loaded = {}
    for patient_id, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        loaded[patient_id] = load_saved_model(path)
    return loaded


def load_demo_data():
    if not DEMO_DATA_PATH.exists():
        raise FileNotFoundError(f"Demo data not found: {DEMO_DATA_PATH}")
    return pd.read_csv(DEMO_DATA_PATH)


def load_demo_model():
    if not DEMO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Demo model not found: {DEMO_MODEL_PATH}")
    return load_saved_model(DEMO_MODEL_PATH)


def extract_doi(text):
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
    return f"doi:{match.group(0).rstrip('.,;') }" if match else None


def split_text(text, chunk_size=1400, overlap=200):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def initialize_rag():
    """Index up to ten local research PDFs in persistent ChromaDB."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="neurocare_research",
        metadata={"hnsw:space": "cosine"},
    )

    pdf_paths = sorted(PAPERS_DIR.glob("*.pdf"))[:10]
    if not pdf_paths:
        return collection, None

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    documents = []
    metadatas = []
    ids = []

    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        citation = extract_doi(text) or pdf_path.stem
        for chunk_number, chunk in enumerate(split_text(text)):
            ids.append(f"{pdf_path.stem}-{chunk_number}")
            documents.append(chunk)
            metadatas.append(
                {
                    "source": pdf_path.name,
                    "citation": citation,
                }
            )

    if documents:
        embeddings = embedder.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return collection, embedder


MODELS = load_models()
DEMO_DATA = load_demo_data()
DEMO_MODEL, DEMO_MODEL_FEATURES = load_demo_model()
RAG_COLLECTION, RAG_EMBEDDER = initialize_rag()


def find_demo_row(patient_id, run_id, window_start_time):
    matches = DEMO_DATA[
        (DEMO_DATA["patient_id"].astype(str) == str(patient_id))
        & (DEMO_DATA["run_id"].astype(str) == str(run_id))
        & (
            np.isclose(
                DEMO_DATA["window_start_time"].astype(float),
                float(window_start_time),
            )
        )
    ]
    if matches.empty:
        raise ValueError("The selected patient/window was not found")
    return matches.iloc[0]


def make_api_array(features):
    missing = [name for name in API_FEATURE_ORDER if name not in features]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    try:
        values = np.array(
            [features[name] for name in API_FEATURE_ORDER],
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("All features must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("All features must be finite numeric values")
    return values


def make_model_array(api_values, model_features):
    values = dict(zip(API_FEATURE_ORDER, api_values))
    if not model_features:
        return api_values.reshape(1, -1)

    model_values = {feature: 0.0 for feature in model_features}
    direct_mappings = {
        "HR_mean": "HR_mean",
        "RMSSD": "HRV",
        "HR_variability": "HRV",
        "EEG_alpha": "EEG_alpha",
        "EEG_beta": "EEG_beta",
        "EEG_delta": "EEG_delta",
        "EEG_theta": "EEG_theta",
        "ACC_energy": "ACC_energy",
    }
    for model_feature, api_feature in direct_mappings.items():
        if model_feature in model_values:
            model_values[model_feature] = values[api_feature]
    if "EEG_ratio" in model_values:
        model_values["EEG_ratio"] = (
            values["EEG_delta"] + values["EEG_theta"]
        ) / (values["EEG_alpha"] + values["EEG_beta"] + 1e-6)
    return np.array(
        [model_values[feature] for feature in model_features],
        dtype=float,
    ).reshape(1, -1)


def top_risk_features(features, model, model_features=None):
    """Rank available API signals by model importance and magnitude."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None or not model_features:
        return list(features)[:3]

    candidates = {
        "HR_mean": ("HR_mean", "high heart rate"),
        "RMSSD": ("HRV", "low HRV"),
        "HR_variability": ("HRV", "low HRV"),
        "EEG_alpha": ("EEG_alpha", "high EEG alpha"),
        "EEG_beta": ("EEG_beta", "high EEG beta"),
        "EEG_delta": ("EEG_delta", "high EEG delta"),
        "EEG_theta": ("EEG_theta", "high EEG theta"),
        "ACC_energy": ("ACC_energy", "high movement energy"),
    }
    scores = {}
    for index, model_feature in enumerate(model_features):
        if model_feature not in candidates:
            continue
        api_name, label = candidates[model_feature]
        value = abs(float(features.get(api_name, 0.0)))
        scores[api_name] = max(scores.get(api_name, 0.0), importances[index] * value)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    labels = {value[0]: value[1] for value in candidates.values()}
    return [labels[api_name] for api_name, _ in ranked[:3]]


def explain_with_rag(top_features):
    query = "Why does " + " and ".join(top_features) + " indicate seizure risk?"
    if RAG_EMBEDDER is None or RAG_COLLECTION.count() == 0:
        return (
            f"The leading signals were {', '.join(top_features)}. "
            "Research-paper retrieval is unavailable until PDFs are added to data/papers.",
            [],
        )

    query_embedding = RAG_EMBEDDER.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()
    results = RAG_COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    citations = list(dict.fromkeys(
        metadata.get("citation") for metadata in metadatas if metadata.get("citation")
    ))
    context = " ".join(documents)[:700]
    explanation = (
        f"The leading signals were {', '.join(top_features)}. "
        f"Relevant research context: {context}"
    )
    return explanation, citations


@app.get("/")
def root():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "app": "NeuroCare",
        "health_endpoint": "GET /health",
        "predict_endpoint": "POST /predict",
        "status": "online",
    })


@app.get("/demo-options")
def demo_options():
    patient_id = request.args.get("patient_id")
    if patient_id:
        patient_rows = DEMO_DATA[DEMO_DATA["patient_id"].astype(str) == patient_id]
        if patient_rows.empty:
            return jsonify({"error": "Unknown patient_id"}), 400
        windows = patient_rows[["run_id", "window_start_time"]].to_dict("records")
        return jsonify({
            "patient_id": patient_id,
            "windows": windows,
        })

    patients = sorted(DEMO_DATA["patient_id"].astype(str).unique())
    return jsonify({"patients": patients})


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
        patient_id = payload.get("patient_id")
        run_id = payload.get("run_id")
        window_start_time = payload.get("window_start_time")
        if patient_id is None or run_id is None or window_start_time is None:
            return jsonify({
                "error": "patient_id, run_id, and window_start_time are required",
            }), 400

        selected_row = find_demo_row(patient_id, run_id, window_start_time)
        missing_features = [
            feature for feature in DEMO_MODEL_FEATURES
            if feature not in selected_row.index
        ]
        if missing_features:
            raise ValueError(
                "Selected dataset row is missing model features: "
                + ", ".join(missing_features)
            )
        try:
            feature_values = np.asarray(
                selected_row[DEMO_MODEL_FEATURES],
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Selected model features must be numeric") from exc
        if feature_values.size != len(DEMO_MODEL_FEATURES):
            raise ValueError(
                f"Expected {len(DEMO_MODEL_FEATURES)} model features, "
                f"got {feature_values.size}"
            )
        if not np.isfinite(feature_values).all():
            raise ValueError("Selected model features must be finite")
        model_values = pd.DataFrame(
            [feature_values],
            columns=DEMO_MODEL_FEATURES,
        )
        probabilities = DEMO_MODEL.predict_proba(model_values)[0]
        class_values = list(DEMO_MODEL.classes_)
        positive_index = class_values.index(1)
        probability = float(probabilities[positive_index])
        prediction = int(DEMO_MODEL.predict(model_values)[0])
        risk_score = probability * 100.0
        return jsonify({
            "risk_score": risk_score,
            "prediction": prediction,
            "label": "High" if prediction == 1 else "Low",
            "recorded_label": int(selected_row["Label"]),
            "patient_id": str(selected_row["patient_id"]),
            "run_id": int(selected_row["run_id"]),
            "window_start_time": float(selected_row["window_start_time"]),
            "feature_count": len(DEMO_MODEL_FEATURES),
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": f"Prediction failed ({type(exc).__name__}): {exc}",
        }), 500


@app.post("/explain")
def explain():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
        risk_score = float(payload["risk_score"])
        features = payload.get("features")
        if not isinstance(features, dict):
            return jsonify({"error": "'features' object is required"}), 400
        api_values = make_api_array(features)
        model, model_features = MODELS[DEFAULT_PATIENT_ID]
        top_features = top_risk_features(features, model, model_features)
        explanation, citations = explain_with_rag(top_features)
        return jsonify({
            "explanation": f"Risk score {risk_score:.1f}. {explanation}",
            "citations": citations,
        })
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid explain request: {exc}"}), 400
    except Exception as exc:
        return jsonify({"error": f"Explanation failed: {exc}"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
