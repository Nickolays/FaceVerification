import torch
import torch.nn.functional as F
from torch import nn
import math


def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    """
    Contrastive loss function.
    
    Args:
        embedding1: Tensor of shape (batch_size, embedding_dim)
        embedding2: Tensor of shape (batch_size, embedding_dim)
        label: Tensor of shape (batch_size,), 1 if same person, 0 if different
        margin: Margin for dissimilar pairs

    Returns:
        loss: Contrastive loss value
    """
    # Compute pairwise L2 distance
    distance = F.pairwise_distance(embedding1, embedding2)
    
    # Contrastive loss formula
    loss = (label) * (distance ** 2) + \
           (1 - label) * (F.relu(margin - distance) ** 2)
    
    return loss.mean()

def bce_distance_loss(embedding1, embedding2, label):
    # Cosine similarity or Euclidean distance
    distance = F.pairwise_distance(embedding1, embedding2)
    
    # You can normalize distance (optional)
    # similarity = torch.exp(-distance)
    similarity = 1 / (1 + distance)  # Like a sigmoid on distance

    # Binary cross-entropy
    loss = F.binary_cross_entropy(similarity, label.float())
    return loss



def arcface_loss(cosine, targ, m=.4):
    """ ArcFace loss function """
    # this prevents nan when a value slightly crosses 1.0 due to numerical error
    cosine = cosine.clip(-1+1e-7, 1-1e-7) 
    # Step 3:
    arcosine = cosine.arccos()
    # Step 4:
    arcosine += F.one_hot(targ, num_classes = 1) * m  # Num_classes
    # Step 5:
    cosine2 = arcosine.cos()
    # Step 6:
    return F.cross_entropy(cosine2, targ)


class PairwiseTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, other, label):
        """
        If label == 1: treat 'other' as positive
        If label == 0: treat 'other' as negative
        """
        # Random shuffle to simulate a negative
        batch_size = anchor.size(0)
        perm = torch.randperm(batch_size)
        shuffled = other[perm]

        # Simulate positive and negative pairs
        positive = other
        negative = torch.where(label.unsqueeze(1) == 1, shuffled, other)

        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)

        # Triplet loss
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()
    

class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.5, scale=30.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, embedding1, embedding2, label):
        """
        Compute arcface-style loss between two embeddings.
        label: 1 for same person, 0 for different.
        """
        # Normalize embeddings
        embedding1 = F.normalize(embedding1)
        embedding2 = F.normalize(embedding2)

        # Cosine similarity
        cosine = (embedding1 * embedding2).sum(dim=1)  # Shape: (batch,)
        
        # Apply margin only when label == 1
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        theta_m = torch.where(label == 1, theta + self.margin, theta)
        logits = torch.cos(theta_m) * self.scale

        # Targets: 1 if same, 0 if different
        loss = F.binary_cross_entropy_with_logits(logits, label.float())
        return loss
    

class MultiNegativeTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negatives):
        """
        anchor: Tensor of shape (batch, embedding_dim)
        positive: Tensor of shape (batch, embedding_dim)
        negatives: Tensor of shape (batch, num_negatives, embedding_dim)
        """
        d_pos = F.pairwise_distance(anchor, positive)  # Shape: (batch,)

        # Compute distance from anchor to each negative
        anchor = anchor.unsqueeze(1)  # (batch, 1, embedding_dim)
        d_neg = F.pairwise_distance(
            anchor.expand_as(negatives).reshape(-1, anchor.size(-1)),
            negatives.reshape(-1, anchor.size(-1))
        ).reshape(anchor.size(0), -1)  # (batch, num_negatives)

        # Use hardest negative
        hardest_d_neg, _ = torch.min(d_neg, dim=1)  # (batch,)

        loss = F.relu(d_pos - hardest_d_neg + self.margin)
        return loss.mean()
    

class ArcFaceClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        weights = F.normalize(self.weight)

        logits = torch.matmul(embeddings, weights.t())  # Cosine similarity
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        theta_margin = theta + self.margin
        logits_with_margin = torch.cos(theta_margin)

        # Scale logits
        logits_scaled = logits_with_margin * self.scale
        loss = F.cross_entropy(logits_scaled, labels)
        return loss