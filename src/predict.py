"""Classify a single image."""

import sys

import numpy as np
from tensorflow import keras

import config
import data


def class_names():
    if config.TRAIN_DIR.is_dir():
        return sorted(p.name for p in config.TRAIN_DIR.iterdir() if p.is_dir())
    return config.CLASSES


def predict(m, image_path):
    names = class_names()
    X = data.load_image_batch(image_path, input_size=m.input_shape[1])

    logits = m.predict(X, verbose=0)[0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()

    return dict(zip(names, probabilities))


def main(model_path, image_path):
    m = keras.models.load_model(model_path)
    scores = predict(m, image_path)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print("prediction:", ranked[0][0])
    for name, p in ranked[:3]:
        print(f"  {name:<12} {p:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python predict.py <model_path> <image_path>")
    main(sys.argv[1], sys.argv[2])
