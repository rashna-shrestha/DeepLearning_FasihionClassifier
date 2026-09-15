from fastapi import FastAPI, UploadFile
from tensorflow import keras
import config
import shutil
import tempfile
import predict


MODEL_PATH = config.MODELS_DIR / "xception_69-0.81.keras"


app = FastAPI(title="Clothing Classifier")
model = keras.models.load_model(MODEL_PATH)

@app.get("/")
def health():
    return {"status": "ok", "model": str(MODEL_PATH)}

@app.post("/predict")
def classify(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        scores = predict.predict(model, tmp.name)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "prediction": ranked[0][0],
        "probabilities": {name: round(float(p), 4) for name, p in ranked},
    }
