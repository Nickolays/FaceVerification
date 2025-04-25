import torch, cv2
from torch import nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


def get_backbone(name:str, pretrained:bool = False):
    """ Returns a backbone model based on the name provided."""
    if pretrained:
        kwargs = {'pretrained': pretrained},
    else:
        kwargs = {'weights': None}
    if name == 'ResNet18':
        # Define backbone (e.g., ResNet18)
        backbone = models.resnet18(**kwargs)
        backbone.fc = nn.Identity()  # Remove classifier
    elif name == 'ResNet50':
        backbone = models.resnet50(**kwargs)
        backbone.fc = nn.Identity()
    elif name == 'MobileNetV2':
        backbone = models.mobilenet_v2(**kwargs)
        backbone.classifier = torch.nn.Sequential(*[
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(1280, 512, bias=False),
            torch.nn.Dropout(p=0.4, inplace=True),
        ])
    elif name == 'EfficientNetB0':
        backbone = models.efficientnet_b0(**kwargs)
        backbone.classifier = torch.nn.Sequential(*[
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(1280, 512, bias=False),
            torch.nn.Dropout(p=0.4, inplace=True),
        ])
    elif name == 'EfficientNetB1':
        backbone = models.efficientnet_b1(**kwargs)
        backbone.classifier = torch.nn.Sequential(*[
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(1280, 512, bias=False),
            torch.nn.Dropout(p=0.5, inplace=True),
        ])
    elif name == 'EfficientNetB2':
        backbone = models.efficientnet_b2(**kwargs)
        backbone.classifier = torch.nn.Sequential(*[
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(1280, 512, bias=False),
            torch.nn.Dropout(p=0.5, inplace=True),
        ])
    elif name == 'EfficientNetB3':
        backbone = models.efficientnet_b3(**kwargs)
        backbone.classifier = torch.nn.Sequential(*[
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(1280, 512, bias=False),
            torch.nn.Dropout(p=0.5, inplace=True),
        ])
    else:
        raise ValueError(f"Unsupported backbone name: {name}")
    
    return backbone

def create_writer(log_dir: str, 
                  model_name: str) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        log_dir (str): Name of experiment.
        model_name (str): Name of model.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")

    tensorboard --logdir=/outputs/... --inspect
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    logdir = os.path.join(log_dir, model_name, timestamp)
    os.makedirs(logdir, exist_ok=True)
        
    print(f"[INFO] Created SummaryWriter, saving to: {logdir}...")
    return SummaryWriter(log_dir=logdir) 

def simple_face_detection(image):
    """
    A simple face detection function using OpenCV's Haar Cascade.
    
    Args:
        image (numpy array): The input image.
    
    Returns:
        The face region of the input image, or None if no faces are detected.
    """
    # Load the face detection cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # If no faces are detected, return None
    if len(faces) == 0:
        return None
    # Otherwise, return the face region
    else:
        return image[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
    
def min_max_scaler(x):
    """
    Min-max scaling function.
    
    Args:
        x (numpy array): The input array to be scaled.
    
    Returns:
        numpy array: The scaled array.
    """
    return (x - x.min()) / (x.max() - x.min())