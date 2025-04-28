# Face Verification

## Project Structure
```
FaceVerification/
├── .dockerignore
├── .env
├── .gitignore
├── Dockerfile                     # CPU-only, production
├── README.md                      # This documentation
├── config_example.yaml            # Hydra config template
├── docker-compose.yml             # CPU-only deployment
├── deploy-gpu/                    # GPU deployment
│   ├── .env
│   ├── Dockerfile
│   └── docker-compose.yml
├── fine_tune.py                   # Fine-tuning script
├── inference.py                   # Single-pair inference CLI
├── inference_snapshot.py          # Snapshot-ensemble evaluation
├── main.py                        # FastAPI app
├── models/                        # Trained model checkpoints
│   └── README.md
├── notebooks/
│   └── test.ipynb
├── prune_quantize.py              # Pruning & quantization utilities
├── requirements-service.txt       # FastAPI requirements
├── requirements.txt               # Core Python dependencies
├── results/                       # Generated artifacts (images, logs)
├── RaspberryPI/                   # Raspberry Pi deployment
│   ├── .env
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements_rpi.txt
├── src/                           # Application source
│   ├── callbacks.py
│   ├── dataset.py
│   ├── helper.py
│   ├── losses.py
│   ├── model.py
│   ├── transforms.py
│   └── utils.py
├── tests/                         # pytest tests
│   ├── test_app.py
│   ├── test_main.py
│   ├── test_image1.jpg
│   └── test_image2.jpg
├── train.py                       # Training entry point (PyTorch Lightning)
└── visualize.py                   # Visualization utilities

```

## Description
Face Verification compares two facial images to verify identity, using deep-learning embeddings and cosine similarity. It includes training, inference, pruning/quantization, and production deployment (CPU, GPU, Raspberry Pi).

### Features
- Pre-trained models for face embeddings.
- Customizable training pipelines.
- Inference: CLI script (inference.py) and FastAPI service (main.py).
- Snapshot Ensembling: Collect multiple models in one training run.
- Pruning & Quantization: Reduce model size for edge devices.
- Deployment: Docker + Docker Compose for CPU, GPU (NVIDIA), and ARM-64 (Raspberry Pi).
- Testing: pytest for data loader, model, API, utilities.

### Requirements
- Python 3.8+
- PyTorch 2.6.0
- OpenCV (headless for server)
- Hydra, TensorBoard (training) 
- FastAPI, Docker, nvidia-docker2 (for GPU) (deploy)
- pytest (test)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/FaceVerification.git
    cd FaceVerification
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    FastAPI service dependencies:
    ```bash
    pip install -r requirements-service.txt
    ```
3. Fill config.yaml

### Usage
1. Training:
```bash
python train.py
```
2. Inference (CLI)
    - For inference
```bash
python inference.py --img1 tests/Adam_Sandler_0001.jpg --img2 tests/Adam_Sandler_0003.jpg --config config.yaml --output_dir results
```

3. FastAPI Deployment
- CPU:
    ```bash
    docker compose up --build
    ```
And open http://localhost:8080/docs
- GPU:
    ```bash
    docker compose -f deploy-gpu/docker-compose.yml up --build
    ```
- For Ruspberry Pi
    ```bash
    # Build the docker image
    docker compose -f RaspberryPI/docker-compose.yml up --build
    ```

### Code Examples
```python
import requests

url = "http://localhost:8080/verify_images"
files = {
    'image1': open('tests/test_image1.jpg','rb'),
    'image2': open('tests/test_image2.jpg','rb'),
}
resp = requests.post(url, files=files)
print("Similarity:", resp.json()["similarity"])
```
Output: similarity

### Tests
For application
```bash
pytest tests/test_app.py --maxfail=1 --disable-warnings -v
```
For FastAPI
```bash
pytest tests/test_main.py --maxfail=1 --disable-warnings -v
```
Inside a running container:
``bash
docker exec -it {CONTAINER_ID} sh
pytest -v --disable-warnings
```


### License
This project is licensed under the MIT License.