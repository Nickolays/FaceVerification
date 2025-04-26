from torchvision import transforms

def get_default_transform(cfg):
    return transforms.Compose([
        transforms.Resize((cfg.transform.resize, cfg.transform.resize)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.transform.normalize_mean,
            std=cfg.transform.normalize_std
        )
    ])