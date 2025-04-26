import io, logging, json
import torch
import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from torchvision import transforms
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.utils import get_backbone, simple_face_detection  # Your backbone loader
from src.transforms import get_default_transform  # Your transform loader

####################################### logger #################################
# logger.remove()
# logger.add(
#     sys.stderr,
#     colorize=True,
#     format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
#     level=10,
# )
# logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

app = FastAPI(title="Face Verification API",
                description="API for face verification using deep learning",
                version="1.0.0")


# app.mount("/static", StaticFiles(directory="src/tests"), name="test_image.jpg")

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.on_event("startup")
# def save_openapi_json():
#     '''This function is used to save the OpenAPI documentation 
#     data of the FastAPI application to a JSON file. 
#     The purpose of saving the OpenAPI documentation data is to have 
#     a permanent and offline record of the API specification, 
#     which can be used for documentation purposes or 
#     to generate client libraries. It is not necessarily needed, 
#     but can be helpful in certain scenarios.'''
#     openapi_data = app.openapi()
#     # Change "openapi.json" to desired filename
#     with open("openapi.json", "w") as file:
#         json.dump(openapi_data, file)

# redirect
# @app.get("/", include_in_schema=False)
# async def redirect():
#     return RedirectResponse("/docs")

@app.get("/")
async def root():
    return {"message": "Face Verification Service is running."}

@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    '''
    It basically sends a GET request to the route & hopes to get a "200"
    response code. Failing to return a 200 response code just enables
    the GitHub Actions to rollback to the last version the project was
    found in a "working condition". It acts as a last line of defense in
    case something goes south.
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'Everything OK!'
    }
    '''
    return {'healthcheck': 'Everything OK!'}

# Load config and model
cfg = OmegaConf.load("config.yaml")
device = torch.device("cpu")

# Setup transform
my_transforms = get_default_transform(cfg)

# Load model backbone
backbone = get_backbone(cfg.backbone.name, pretrained=False)
backbone.load_state_dict(torch.load(cfg.infer.model_path, map_location=device))
backbone.eval().to(device)

# Pydantic output validation
class SimilarityResponse(BaseModel):
    similarity: float


def preprocess(file: UploadFile):
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    face = simple_face_detection(img)
    if face is None:
        raise HTTPException(status_code=400, detail="Face not detected.")
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return my_transforms(transforms.ToPILImage()(face)).unsqueeze(0)

@app.post("/verify_images", response_model=SimilarityResponse)
async def verify_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    try:
        img1_tensor = preprocess(await image1.read())
        img2_tensor = preprocess(await image2.read())

        with torch.no_grad():
            emb1, emb2 = backbone(img1_tensor), backbone(img2_tensor)
            similarity = F.cosine_similarity(emb1, emb2).item()

        return {"similarity": similarity}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# @app.post("/verify_images", response_model=SimilarityResponse)
# async def verify_images(
#     image1: UploadFile = File(...),
#     image2: UploadFile = File(...)
# ):
    # try:
    #     # Read and decode images
    #     image_bytes1 = await image1.read()
    #     image_bytes2 = await image2.read()

    #     nparr1 = np.frombuffer(image_bytes1, np.uint8)
    #     nparr2 = np.frombuffer(image_bytes2, np.uint8)

    #     img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    #     img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    #     # Face detection
    #     face1 = simple_face_detection(img1)
    #     face2 = simple_face_detection(img2)

    #     if face1 is None or face2 is None:
    #         raise HTTPException(status_code=400, detail="Face not detected in one or both images.")

    #     # Transform
    #     face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
    #     face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)

    #     face1 = my_transforms(transforms.ToPILImage()(face1)).unsqueeze(0)
    #     face2 = my_transforms(transforms.ToPILImage()(face2)).unsqueeze(0)

    #     # Embed
    #     with torch.no_grad():
    #         emb1 = backbone(face1)
    #         emb2 = backbone(face2)

    #     # Cosine similarity
    #     similarity = F.cosine_similarity(emb1, emb2).item()

    #     return {"similarity": similarity}
    
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    
# @app.get("/helthcheck")
# async def healthcheck():
#     return JSONResponse(content={"status": "ok"}, status_code=200)

# Example usage with curl
# curl -X POST "http://localhost:8000/verify_images" \
#   -F "image1=@tests/Adam_Sandler_0001.jpg" \
#   -F "image2=@tests/Adam_Sandler_0003.jpg"