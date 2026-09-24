# Fashion Clothing Classifier

A lightweight deep learning project for classifying clothing images into 10 categories using TensorFlow/Keras and transfer learning.

## Overview

This project trains a convolutional neural network based on the Xception architecture and serves it through a FastAPI API for image classification. It is designed for a small clothing dataset and is suitable for local experimentation, training, evaluation, and simple deployment.

### Supported classes

- dress
- hat
- longsleeve
- outwear
- pants
- shirt
- shoes
- shorts
- skirt
- t-shirt

## Features

- Transfer learning with Xception pretrained on ImageNet
- Training pipeline for clothing image classification
- Validation and test evaluation scripts
- Single-image prediction utilities
- REST API for inference via FastAPI
- Docker support for containerized deployment
- Training history plots saved locally

## Tech stack

- Python 3.12
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Pillow
- NumPy
- Docker

## Project structure

```text
.
├── Dockerfile
├── requirements.txt
├── README.md
├── clothing-dataset-small/
│   ├── train/
│   ├── validation/
│   └── test/
├── notebook/
│   └── notebook.ipynb
├── src/
│   ├── app.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   ├── models/
│   └── plots/
└── .venv/
```

## Dataset

The project expects image folders under `clothing-dataset-small/` with one class per subfolder. Each split is organized as:

```text
clothing-dataset-small/
├── train/
│   ├── dress/
│   ├── hat/
│   ├── longsleeve/
│   ├── outwear/
│   ├── pants/
│   ├── shirt/
│   ├── shoes/
│   ├── shorts/
│   ├── skirt/
│   └── t-shirt/
├── validation/
├── test/
└── ...
```

The image data is loaded with Keras `ImageDataGenerator` and preprocessed using Xception preprocessing.

## Configuration

Main settings are defined in `src/config.py`:

- dataset root: `clothing-dataset-small`
- train path: `clothing-dataset-small/train`
- validation path: `clothing-dataset-small/validation`
- test path: `clothing-dataset-small/test`
- model output directory: `src/models`
- plot output directory: `src/plots`
- input size: `150 x 150`
- batch size: `32`
- learning rate: `0.001`
- epochs: `80`

## Quick start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Training

Run the training script from the project root:

```bash
cd src
python train.py
```

### What the training script does

- builds the train and validation datasets
- applies augmentation to improve robustness
- loads a frozen Xception base model
- adds a global average pooling layer and dense classification head
- trains and saves the best model checkpoint
- saves a training accuracy plot to `src/plots/history.png`

The project saves checkpoints in the format:

```text
xception_{epoch:02d}-{val_accuracy:.2f}.keras
```

## Evaluation

To evaluate a saved model on the test set:

```bash
python src/evaluate.py src/models/xception_69-0.81.keras
```

This prints the test loss and test accuracy.

## Prediction from the command line

To classify a single image:

```bash
python src/predict.py src/models/xception_69-0.81.keras path/to/image.jpg
```

The script reports the predicted class and the top probability scores.

## API usage

The project exposes a FastAPI app in `src/app.py`.

### Start the API

```bash
cd src
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /` — health check
- `POST /predict` — upload an image and get the predicted class with probabilities

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@example.jpg"
```

### Example response

```json
{
  "prediction": "shirt",
  "probabilities": {
    "dress": 0.0123,
    "hat": 0.0012,
    "longsleeve": 0.0456,
    "outwear": 0.0098,
    "pants": 0.0101,
    "shirt": 0.8824,
    "shoes": 0.0031,
    "shorts": 0.0205,
    "skirt": 0.0117,
    "t-shirt": 0.0023
  }
}
```

## Docker

Build the Docker image:

```bash
docker build -t clothing-classifier .
```

Run the container:

```bash
docker run -p 8000:8000 clothing-classifier
```

This starts the FastAPI service on port `8000`.

## Model notes

- The base model uses Xception pretrained on ImageNet.
- The top layers are removed and replaced with a custom classifier.
- The model is trained for a relatively small clothing dataset, so accuracy depends on dataset quality and labeling consistency.
- Existing trained checkpoints are stored in `src/models/` and can be reused or replaced with new training results.

## Possible next improvements

- add a small frontend for uploading images
- add confusion matrix and precision/recall reporting
- retrain on a larger dataset for better generalization
- add logging, monitoring, and deployment configuration
- add automated tests for the training and prediction pipeline

## License

This project does not currently include a license file. If you plan to share or distribute it publicly, consider adding an appropriate open-source license.
