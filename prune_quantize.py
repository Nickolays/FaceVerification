import torch
import torch.nn.utils.prune as prune
import torch.quantization
import time
import os
from omegaconf import OmegaConf
from copy import deepcopy
from PIL import Image

from src.utils import get_backbone, fuse_resnet, fuse_mobilenetv2, fuse_efficientnet
from src.transforms import get_default_transform

def print_model_size(model, filename="temp.pth"):
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / 1e6
    print(f"Model size: {size_mb:.2f} MB")
    os.remove(filename)

def benchmark_inference(model, input_size=(1, 3, 112, 112), device='cpu'):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size).to(device)
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        end = time.time()
    avg_time = (end - start) / 100
    print(f"Avg inference time per image: {avg_time * 1000:.2f} ms")

# def prune_model(model, amount=0.3):
#     """
#     Apply global unstructured pruning to the Conv2d layers of a model.

#     Args:
#         model (torch.nn.Module): The model to prune.
#         amount (float): The proportion of connections to prune.

#     Returns:
#         torch.nn.Module: The pruned model.
#     """
#     # Prepare a list of parameters to prune
#     parameters_to_prune = []
#     for module in model.modules():
#         # Target Conv2d layer weights for pruning
#         if isinstance(module, torch.nn.Conv2d):
#             parameters_to_prune.append((module, 'weight'))

#     # Apply global unstructured L1 pruning
#     model = prune.global_unstructured(
#         parameters_to_prune,
#         pruning_method=prune.L1Unstructured,
#         amount=amount,
#     )

#     return model

def prune_model(model, amount=0.3):
    """
    Apply global unstructured pruning to the Conv2d layers of a model.

    Args:
        model (torch.nn.Module): The model to prune.
        amount (float): The proportion of connections to prune.

    Returns:
        torch.nn.Module: The pruned model.
    """
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    # Apply global unstructured L1 pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Optionally, remove pruning reparameterization to make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    return model

def quantize_model(model, input, method="dynamic"):
    model.eval()
    if method == "dynamic":
        return torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    elif method == "static":
        model.fuse_model() if hasattr(model, 'fuse_model') else None
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        for _ in range(10):
            model(input)
        torch.quantization.convert(model, inplace=True)
        return model
    else:
        raise ValueError(f"Unknown quantization method: {method}")

if __name__ == "__main__":
    # === Load config ===
    cfg = OmegaConf.load("config.yaml")
    backbone_name = cfg.backbone.name
    pretrained = cfg.backbone.pretrained
    input_resize = cfg.transform.resize

    # === Load and fuse backbone ===
    model = get_backbone(backbone_name, pretrained=False)
    model.load_state_dict(torch.load(cfg.infer.model_path, 
                                    map_location='cpu', 
                                    weights_only=False))

    print("Before pruning:")
    print_model_size(model)
    benchmark_inference(model, input_size=(32, 3, input_resize, input_resize))

    transform = get_default_transform(cfg)
    img1 = transform(Image.open("tests/Adam_Sandler_0001.jpg").convert("RGB")).unsqueeze(0)

    # === Prune model ===
    # pr_model = deepcopy(model)
    model = prune_model(model, amount=0.3)
    print("\n== Before Quantization ==")
    print_model_size(model)
    benchmark_inference(model, input_size=(32, 3, input_resize, input_resize))

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"Conv2d weights nonzero count: {(module.weight != 0).sum().item()}")

    model.eval()  # <-- this line is critical for fusion!
    if "ResNet" in backbone_name:
        model = fuse_resnet(model)
    elif "model" in backbone_name:
        model = fuse_mobilenetv2(model)
    elif "EfficientNet" in backbone_name:
        model = fuse_efficientnet(model)

    # === Quantize model ===
    quantized_model = quantize_model(model, img1, method="dynamic")

    print("\n== After Quantization ==")
    print_model_size(quantized_model, filename="quantized_model.pth")
    benchmark_inference(quantized_model, input_size=(32, 3, input_resize, input_resize))

    # Save quantized model
    torch.save(quantized_model.state_dict(), "models/quantized_backbone.pth")
    print("Quantized model saved to models/quantized_backbone.pth")
