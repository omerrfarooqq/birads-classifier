import torch
from utils.metrics import plot_confusion_matrix
import os
import shutil

def evaluate(model, loader):
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.cuda()
            labels = labels.cuda()

            preds = model(imgs)
            _, pred_class = preds.max(1)

            preds_list.extend(pred_class.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    return 0, preds_list, labels_list
