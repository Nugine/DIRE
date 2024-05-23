import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from utils.config import CONFIGCLASS
from utils.utils import to_cuda


def get_val_cfg(cfg: CONFIGCLASS, split="val", copy=True):
    if copy:
        from copy import deepcopy

        val_cfg = deepcopy(cfg)
    else:
        val_cfg = cfg
    val_cfg.dataset_root = os.path.join(val_cfg.dataset_root, split)
    val_cfg.datasets = cfg.datasets_test
    val_cfg.isTrain = False
    # val_cfg.aug_resize = False
    # val_cfg.aug_crop = False
    val_cfg.aug_flip = False
    val_cfg.serial_batches = True
    val_cfg.jpg_method = ["pil"]
    # Currently assumes jpg_prob, blur_prob 0 or 1
    if len(val_cfg.blur_sig) == 2:
        b_sig = val_cfg.blur_sig
        val_cfg.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_cfg.jpg_qual) != 1:
        j_qual = val_cfg.jpg_qual
        val_cfg.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_cfg


def validate(model: nn.Module, cfg: CONFIGCLASS):
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    from utils.datasets import create_dataloader

    data_loader = create_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:
            img, label, meta = data if len(data) == 3 else (*data, None)
            in_tens = to_cuda(img, device)
            meta = to_cuda(meta, device)
            predict = model(in_tens, meta).sigmoid()
            y_pred.extend(predict.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    save_path = os.path.join(cfg.exp_dir, "validate.npz")
    np.savez(save_path, y_true=y_true, y_pred=y_pred)

    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred > 0.5).ravel()
    auc = roc_auc_score(y_true, y_pred)
    results = {
        "ACC": acc,
        "AP": ap,
        "R_ACC": r_acc,
        "F_ACC": f_acc,
        "ConfusionMatrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "AUC": auc,
    }
    return results


def vis_gradcam(model: nn.Module, cfg: CONFIGCLASS):
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import cv2
    from PIL import Image
    import random
    from shutil import copyfile

    from utils.datasets import create_dataloader, get_transform

    data_loader = create_dataloader(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    validate = np.load(os.path.join(cfg.exp_dir, "validate.npz"))
    y_true, y_pred = validate["y_true"], validate["y_pred"]

    indices = [
        idx for idx in range(len(y_true)) if y_true[idx] == 1 and y_pred[idx] > 0.95
    ]
    input_tensor_idx = random.choice(indices)
    input_tensor = data_loader.dataset[input_tensor_idx][0].unsqueeze(0).to(device)

    print(f"pred = {y_pred[input_tensor_idx]:.5f}, true = {y_true[input_tensor_idx]}")

    target_layers = [model.layer4[-1]]
    # cam = GradCAM(model=model, target_layers=target_layers)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]

    image_path = data_loader.dataset.datasets[0].imgs[input_tensor_idx][0]
    print("original image path:", image_path)

    rgb_image = Image.open(image_path)
    rgb_image = (
        get_transform(cfg, visualizing=True)(rgb_image).numpy().transpose(1, 2, 0)
    )

    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    cam_path = os.path.join(cfg.exp_dir, "gradcam.jpg")
    print("gradcam saved at:", cam_path)
    cv2.imwrite(cam_path, cam_image)

    copyfile(image_path, os.path.join(cfg.exp_dir, "original.jpg"))
