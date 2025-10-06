# metrics.py

import torch
import torch.nn.functional as F

def dice_per_class(logits, targets, num_classes, smooth=1e-5):
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    dice_scores = []

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    return dice_scores

def sensitivity(logits, targets, cls, smooth=1e-6):
    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    TP = ((preds == cls) & (targets == cls)).sum().float()
    FN = ((preds != cls) & (targets == cls)).sum().float()
    return (TP + smooth) / (TP + FN + smooth)

def specificity(logits, targets, cls, smooth=1e-6):
    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    TN = ((preds != cls) & (targets != cls)).sum().float()
    FP = ((preds == cls) & (targets != cls)).sum().float()
    return (TN + smooth) / (TN + FP + smooth)

def ppv(logits, targets, cls, smooth=1e-6):
    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    TP = ((preds == cls) & (targets == cls)).sum().float()
    FP = ((preds == cls) & (targets != cls)).sum().float()
    return (TP + smooth) / (TP + FP + smooth)

def miou(logits, targets, num_classes, smooth=1e-6):
    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    ious = []

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou.item())

    return ious
