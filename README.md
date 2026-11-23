# Pneumonia CNN

A PyTorch chest X-ray classifier with a FastAPI prediction endpoint and MLflow experiment tracking. Run the commands below from the repository root. This is a learning project, not a medical diagnostic tool.

## Run the API locally

You need Python 3.14, [uv](https://docs.astral.sh/uv/), and `models/model.pt`. The checkpoint is required for inference; the dataset is not.

```bash
uv sync
uv run uvicorn pneunomia_cnn.api:app --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860/docs> to try the API in your browser, or send a JPEG or PNG image with `curl`:

```bash
curl -F "file=@/path/to/chest-xray.jpeg" http://127.0.0.1:7860/predict
```

`GET /health` checks that the service is running. `POST /predict` returns a `NORMAL` or `PNEUMONIA` label and a `pneumonia_score`. The label uses a score threshold of 0.5; the score is not a calibrated clinical probability.

## Run with Docker

Docker also requires `models/model.pt` before the image can be built. The image serves the API on port 7860.

```bash
docker build -t pneumonia-cnn .
docker run --rm -p 7860:7860 pneumonia-cnn
```

The API docs are then at <http://127.0.0.1:7860/docs>. Rebuild the image if you retrain the model and want it to use the new checkpoint.

## Download the dataset

Evaluation and training require the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Install and authenticate the Kaggle CLI, then run:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
```

The scripts expect this layout:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The `data/` directory is ignored by Git.

## Evaluate the checkpoint

With the dataset and `models/model.pt` in place, run:

```bash
uv run python -m pneunomia_cnn.evaluate
```

This prints accuracy and the numbers of correct and incorrect predictions on `data/chest_xray/test` using the 0.5 threshold.

## Train and view runs in MLflow

Start an MLflow server in one terminal:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Then, from the repository root in another terminal, run:

```bash
uv run python -m pneunomia_cnn.train
```

View the `pneumonia-cnn` experiment at <http://127.0.0.1:5000>. Training uses an 85/15 split of `data/chest_xray/train`, logs metrics for 10 epochs, and writes a new checkpoint to `models/model.pt`. The training script expects the MLflow server at `http://127.0.0.1:5000`.
