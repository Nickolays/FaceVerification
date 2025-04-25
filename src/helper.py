import torch, os
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F


def evaluate_model(model, dataloader, threshold=0.5):
    """
    Evaluate a single model on the given dataloader.
    Returns predictions, targets, and similarity scores.
    """
    model.eval()
    preds_all, targets_all, scores_all = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x1, x2, target = batch['face1'], batch['face2'], batch['target']
            emb1, emb2 = model(x1.cuda()), model(x2.cuda())
            similarity = F.cosine_similarity(emb1, emb2)
            preds = (similarity > threshold).long()
            preds_all.append(preds.cpu())
            targets_all.append(target.cpu())
            scores_all.append(similarity.cpu())

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)
    scores_all = torch.cat(scores_all)

    acc = accuracy_score(targets_all, preds_all)
    try:
        roc = roc_auc_score(targets_all, scores_all)
    except:
        roc = None
    return acc, roc, preds_all, targets_all, scores_all

def load_snapshot_models(folder_path, model):
    """
    Load all .pth models from the specified folder and return a list of FaceVerificationModel instances.
    
    Args:
        folder_path (str): The path to the directory containing the .pth files.
        cfg (DictConfig): The configuration object containing model parameters.
        backbone: The backbone model to use for loading the checkpoints.
        transform: The data transformation to apply to the model.
    
    Returns:
        list: A list of FaceVerificationModel instances loaded from the checkpoints.
    """
    models = []
    # Iterate through sorted files in the folder
    for file in sorted(os.listdir(folder_path)):
        # Check if the file is a checkpoint file
        if file.endswith(".pth"):
            path = os.path.join(folder_path, file)
            # Load the model from checkpoint
            # model = torch.load(path, map_location=torch.device('cuda'))
            model.load_state_dict(torch.load(path, 
                                            map_location='cpu', 
                                            weights_only=False))
            # Move model to GPU
            model.cuda()
            # Append to the models list
            models.append(model)
    return models

def evaluate_snapshot_ensemble(models, dataloader, threshold=0.5):
    """
    Performs ensemble evaluation using the average similarity from all snapshot models.
    """
    all_scores = []

    for model in models:
        model.eval()
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                x1, x2 = batch['face1'], batch['face2']
                emb1 = model(x1.cuda())
                emb2 = model(x2.cuda())
                sim = F.cosine_similarity(emb1, emb2).cpu()
                scores.append(sim)
        all_scores.append(torch.cat(scores))

    # Average all similarity scores across models
    ensemble_scores = torch.stack(all_scores).mean(dim=0)
    targets = torch.cat([batch['target'] for batch in dataloader])
    preds = (ensemble_scores > threshold).long()

    acc = accuracy_score(targets, preds)
    try:
        roc = roc_auc_score(targets, ensemble_scores)
    except:
        roc = None
    return acc, roc


# If we want to use for pytorch lightning in .cpkt format
# def load_snapshot_models(folder_path, cfg, backbone, transform):
#     """
#     Load all .ckpt models from the specified folder and return a list of FaceVerificationModel instances.
#     TODO: Add .cpkt loading 
#     Args:
#         folder_path (str): The path to the directory containing the .pth files.
#         cfg (DictConfig): The configuration object containing model parameters.
#         backbone: The backbone model to use for loading the checkpoints.
#         transform: The data transformation to apply to the model.
    
#     Returns:
#         list: A list of FaceVerificationModel instances loaded from the checkpoints.
#     Load all .pth models from the folder and return list of models instances.
#     """
#     models = []
#     # Iterate through sorted files in the folder
#     for file in sorted(os.listdir(folder_path)):
#         # Check if the file is a checkpoint file
#         if file.endswith(".ckpt"):
#             path = os.path.join(folder_path, file)
#             # Load the model from checkpoint
#             model = FaceVerificationModel.load_from_checkpoint(
#                 path, config=cfg, backbone=backbone, transform=transform
#             )
#             # Move model to GPU
#             model.cuda()
#             # Append to the models list
#             models.append(model)
#     return models