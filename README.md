# Renal Tumor Classification — Kidney CT Scan Classification Pipeline

A machine learning pipeline that classifies kidney CT scan images as **Normal** or **Tumor**. It covers the full process from raw data to a deployable web application: data ingestion, transfer-learning model training (VGG16), evaluation, experiment tracking, pipeline versioning, and deployment.

**What it includes:**
- Trains a transfer-learning CNN (VGG16) on kidney CT scan images
- Tracks experiments (params, metrics, model artifacts) with MLflow, hosted on DagsHub
- Versions data and pipeline outputs with DVC, so the pipeline only reruns stages that actually changed
- Serves real-time predictions through a Flask web app with an upload-and-predict UI
- Builds and deploys automatically via GitHub Actions → Docker → AWS ECR/EC2 on every push to `main`

---

## 1. What This Project Does

This project trains an image classification model that looks at a kidney CT scan and predicts whether it shows a **normal kidney** or a **tumor**. It covers everything from raw data to a deployable web app:

1. **Data Ingestion** — downloads a zipped kidney CT scan image dataset and extracts it.
2. **Prepare Base Model** — loads a pretrained **VGG16** model (trained on ImageNet), removes its top classification layer, and adds a new output layer for our 2 classes (Normal / Tumor). This is **transfer learning** — reusing a model that already knows how to recognize general image features, instead of training from scratch.
3. **Model Training** — fine-tunes the modified VGG16 model on the kidney CT scan dataset, using image augmentation for better generalization.
4. **Model Evaluation** — evaluates the trained model's loss and accuracy on a validation split, and optionally logs the results to MLflow (hosted on DagsHub) for experiment tracking.
5. **Web App** — a Flask app where a user can upload a kidney CT scan image and get an instant prediction (Normal / Tumor).

The whole training pipeline can also be re-triggered directly from the web app's `/train` endpoint, and every step's inputs/outputs are wired together with DVC so the pipeline only reruns the stages that actually changed.

---

## 2. Tech Stack

| Purpose | Tool |
|---|---|
| Model | TensorFlow / Keras (VGG16, transfer learning) |
| Pipeline orchestration | Custom Python pipeline + DVC |
| Experiment tracking | MLflow (hosted via DagsHub) |
| Web app / API | Flask |
| Data/model versioning | DVC (Data Version Control) |
| CI/CD | GitHub Actions → AWS ECR → AWS EC2 (Docker) |
| Config management | YAML (`config.yaml`, `params.yaml`) + `python-box` |

---

## 3. How to Run This Project Locally

### Step 1 — Clone the repo
```bash
git clone https://github.com/<your-username>/Renal-Tumor-Classification-.git
cd Renal-Tumor-Classification-
```

### Step 2 — Create and activate a virtual environment (Python 3.8)
```bash
conda create -n cnncls python=3.8 -y
conda activate cnncls
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the full training pipeline
```bash
python main.py
```
This downloads the dataset, prepares the base VGG16 model, trains it, and evaluates it — producing `artifacts/training/model.h5` and `scores.json`.

### Step 5 — Get the model into place for the web app
The Flask app loads its model from `model/model.h5` (not `artifacts/training/model.h5`). After training:
```bash
mkdir model
copy artifacts\training\model.h5 model\model.h5     # Windows
# cp artifacts/training/model.h5 model/model.h5     # macOS/Linux
```

### Step 6 — Run the web app
```bash
python app.py
```
Open your browser at **http://localhost:8080**

- Click **Predict** after uploading a kidney CT scan image to get a classification.
- Click **Train** to re-run the entire training pipeline from the browser (equivalent to running `python main.py`).

---

## 4. Running via DVC (optional, reproducible pipeline)

Instead of `python main.py`, you can let DVC manage and cache each stage, so it only reruns stages whose inputs actually changed:
```bash
dvc repro
```
Visualize the pipeline graph:
```bash
dvc dag
```

---

## 5. MLflow / DagsHub Experiment Tracking (optional)

To log training runs (parameters, metrics, and the trained model itself) to a DagsHub-hosted MLflow server:
```python
import dagshub
dagshub.init(repo_owner='<your-dagshub-username>', repo_name='<your-repo-name>', mlflow=True)
```
Run this once at the top of a notebook or script before calling `evaluation.log_into_mlflow()`. It handles authentication and sets the required environment variables automatically.

---

## 6. Deployment (AWS CI/CD via GitHub Actions)

`.github/workflows/main.yaml` automates deployment on every push to `main`:

1. **Continuous Integration** — checks out the code (linting/tests are placeholders here, ready to be filled in).
2. **Continuous Delivery** — builds a Docker image from the `Dockerfile`, pushes it to **AWS ECR** (Elastic Container Registry).
3. A self-hosted GitHub Actions runner on an **AWS EC2** instance then pulls and runs the latest image, exposing the Flask app.

To use this yourself, you'd need to set these as GitHub repo secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ECR_LOGIN_URI`, `ECR_REPOSITORY_NAME`.

---

## 7. Dataset

Kidney CT scan images, labeled as **Normal** or **Tumor**, downloaded automatically at pipeline runtime from the `source_URL` specified in `config/config.yaml`.

---

## 8. Model Performance

Latest evaluation metrics are tracked in `scores.json` (updated automatically each time `main.py` / `dvc repro` is run):
```json
{
    "loss": ...,
    "accuracy": ...
}
```

---

## 9. Author

**Likhitha Guduru**
GitHub: [likhitha09guduru](https://github.com/likhitha09guduru)

---

## 10. Project Structure

```
Renal-Tumor-Classification-/
│
├── .github/workflows/main.yaml   # CI/CD pipeline: build & push Docker image to AWS on every push to main
├── .dvc/                         # DVC internal config (tracks pipeline artifacts)
├── config/
│   └── config.yaml               # Paths and URLs used by every pipeline stage (data source, artifact paths, model paths)
├── params.yaml                   # Model hyperparameters (image size, batch size, epochs, learning rate, etc.)
├── dvc.yaml                      # Defines the DVC pipeline: stages, dependencies, and outputs
├── dvc.lock                      # Auto-generated snapshot of the exact DVC pipeline state (do not edit manually)
├── scores.json                   # Latest evaluation metrics (loss, accuracy) — tracked by DVC as a metric file
├── main.py                       # Runs the full pipeline end-to-end: ingestion → base model → training → evaluation
├── app.py                        # Flask web app: serves the UI, and exposes /train and /predict endpoints
├── setup.py                      # Makes src/cnnClassifier installable as a local Python package (pip install -e .)
├── requirements.txt              # All Python dependencies needed to run this project
├── Dockerfile                    # Builds a container image that runs app.py (used for AWS deployment)
├── template.py                   # One-time script used to scaffold this project's folder/file structure
├── templates/
│   └── index.html                # Front-end UI: upload an image, trigger prediction, trigger training
│
├── Research/                     # Jupyter notebooks used to prototype each pipeline stage before refactoring into src/
│   ├── 1_data_ingestion.ipynb
│   ├── 2_base_model.ipynb
│   ├── 3_model_training.ipynb
│   ├── 4_model_evaluation.ipynb
│   └── trials.ipynb
│
├── model/                        # (not committed by default — trained locally or fetched separately)
│   └── model.h5                  # The trained model file the Flask app loads for predictions
│
├── artifacts/                    # (gitignored) All pipeline outputs: downloaded dataset, base model, trained model
│
└── src/cnnClassifier/            # The core installable Python package — all real logic lives here
    ├── __init__.py                    # Package-wide logger setup (writes to logs/running_logs.log and stdout)
    ├── constants/
    │   └── __init__.py                # File paths to config.yaml and params.yaml
    ├── utils/
    │   └── common.py                  # Shared helper functions: read_yaml, create_directories, save_json,
    │                                   #   load_json, save_bin, load_bin, get_size, decodeImage, encodeImageIntoBase64
    ├── entity/
    │   └── config_entity.py           # Typed "config" dataclasses (e.g. DataIngestionConfig, TrainingConfig)
    │                                   #   — defines exactly what settings each pipeline stage expects
    ├── config/
    │   └── configuration.py           # ConfigurationManager: reads config.yaml + params.yaml and builds the
    │                                   #   typed config objects (from entity/) that each component needs
    ├── components/                    # The actual logic for each pipeline stage
    │   ├── data_ingestion.py              # Downloads the dataset zip from Google Drive and extracts it
    │   ├── base_model.py                  # Builds VGG16 base model, adds a custom classification head
    │   ├── model_training.py              # Trains the model with Keras ImageDataGenerator (with augmentation)
    │   └── model_evaluation.py            # Evaluates the trained model, saves scores.json, logs to MLflow/DagsHub
    ├── pipeline/                      # Wrappers that call the components above, in order, for each DVC stage
    │   ├── stage1_data_ingestion.py       # Runs DataIngestion component
    │   ├── stage2_base_model.py           # Runs PrepareBaseModel component
    │   ├── stage3_model_training.py       # Runs Training component
    │   ├── stage4_model_evaluation.py     # Runs Evaluation component
    │   └── prediction.py                  # PredictionPipeline — loads model/model.h5 and predicts on a single image
    └── components/__init__.py, etc.   # Package init files (make each folder importable)
```

### How the pieces connect

```
config.yaml + params.yaml
        │
        ▼
ConfigurationManager (config/configuration.py)
        │   builds typed config objects using entity/config_entity.py
        ▼
Components (components/*.py)
        │   the actual data ingestion / training / evaluation logic
        ▼
Pipeline stages (pipeline/stage*.py)
        │   call components in the right order for one DVC stage
        ▼
main.py  ──────────────► runs all 4 stages back-to-back
        │
        ▼
artifacts/  (dataset, base model, trained model.h5)
        │
        ▼
app.py  ──────────────► loads model/model.h5 via pipeline/prediction.py
        │                to serve real-time predictions
        ▼
templates/index.html (the UI the user interacts with)
```
