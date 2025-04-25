# Face Verification

## Project Structure
```
FaceVerification/
├── data/               # Directory for datasets
├── models/             # Pre-trained and custom models
├── notebooks/          # .ipynb notebooks
├── src/                # Python scripts for training and evaluation
├── tests/              # pytest
├── train.py            # Main function to train the FaceVerificationModel using PyTorch Lightning, using config
├── inference.py        # Script for inference model
├── config.yaml         # Hydra config file
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Description
Face Verification is a project designed to verify the identity of individuals by comparing facial images. It leverages state-of-the-art deep learning techniques for facial recognition and verification tasks.

### Features
- Pre-trained models for face embeddings.
- Customizable training pipelines.
- Support for multiple datasets.
- Easy-to-use scripts for evaluation and testing.

### Requirements
- Python 3.8 or higher
- PyTorch
- OpenCV for image processing
- Hydra, tensorboard, pytest

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

### Usage
Run the main script to verify faces:
```bash
python inference.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```
For instance
```bash
python inference.py --img1 tests/Adam_Sandler_0001.jpg --img2 tests/Adam_Sandler_0003.jpg --config config.yaml --output_dir results
```

# TODO:
1. Tensorboard logging                         (Done)
1.1 Snapshot Ensembling and Gradient Clipping  (Done)
2. Visualize, compare my photos                (Done)
2. FineTuning                                  (Done)
3. Model size reduction                        ()
4. Docker + FastAPI
5. Mini PC 
6. GPU deploy
7. Full project (Face Detection + Face Recognition)


### License
This project is licensed under the MIT License.