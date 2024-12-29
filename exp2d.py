import os
import numpy as np
import dagshub
import mlflow
dagshub.init(repo_owner='huytrnq', repo_name='BrainSegmentation', mlflow=True)
# Start MLflow tracking
mlflow.start_run(run_name="EfficientNet-UNet")

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from utils.loss import DiceLoss, DiceCrossEntropyLoss, DiceFocalLoss
from utils.dataset import BrainMRISliceDataset
from utils.utils import train, validate
from utils.metric import MetricsMonitor, dice_coefficient, dice_score_3d

#################### Hyperparameters ####################
ROOT_DIR = './Data/'
BATCH_SIZE = 16
EPOCHS = 300
DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0
LR = 0.001

if __name__ == '__main__':    
    print(f"Using device: {DEVICE}")
    #################### DataLoaders #################### 

    # Augmentation pipeline
    train_transform = A.Compose([
        # Spatial Transformations
        A.OneOf([
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.2), rotate=(-30, 30), shear=(-15, 15), p=1.0),  # Scaling, Rotation, Shearing
            A.ElasticTransform(alpha=1.0, sigma=50.0, p=1.0),  # Elastic deformation
        ], p=0.5),  # 50% chance to apply one of the spatial transforms

        # Intensity Perturbations
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),  # Gaussian Blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),  # Gaussian Noise
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=1.0),  # Brightness Multiplicative Transform
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),  # Brightness and Contrast
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),  # Gamma Transform
        ], p=0.5),  # 50% chance to apply one of the intensity perturbations

        # Other Transformations
        A.OneOf([
            A.HorizontalFlip(p=1.0),  # Mirroring
            A.VerticalFlip(p=1.0),  # Mirroring
        ], p=0.5),

        # Normalize and convert to tensors
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize to ImageNet stats
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})  # Include mask augmentation

    test_transform = A.Compose([
        # A.Resize(256, 256),
        # A.LongestMaxSize(max_size=256),  # Resize the smallest side to 256, keeping the aspect ratio
        # A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0),  # Pad to a square image
        # A.Normalize(normalization="min_max", p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

    train_dataset = BrainMRISliceDataset(os.path.join(ROOT_DIR, 'train'), slice_axis=2, transform=train_transform, cache=True, ignore_background=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_dataset = BrainMRISliceDataset(os.path.join(ROOT_DIR, 'val'), slice_axis=2, transform=test_transform, cache=True, ignore_background=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    #################### Model ####################
    
    model = smp.Segformer(
        encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=4,                      # model output channels (number of classes in your dataset)
    )
    model = model.to(DEVICE)
    
    #################### Loss, Optimizer, Scheduler ####################
    criterion = DiceCrossEntropyLoss(dice_weight=0.5, ce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    
    #################### Training ####################
    # Monitors
    train_monitor = MetricsMonitor(metrics=["loss", "dice_score"])
    val_monitor = MetricsMonitor(
        metrics=["loss", "dice_score"], patience=20, mode="max"
    )
    test_monitor = MetricsMonitor(metrics=["loss", "dice_score"])
    
    #################### MLflow ####################
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("model", model.__class__.__name__)
    mlflow.log_param("optimizer", optimizer.__class__.__name__)
    mlflow.log_param("scheduler", scheduler.__class__.__name__)
    mlflow.log_param("criterion", criterion.__class__.__name__)    

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        # Train the model
        train_metric_dict = train(model, train_loader, criterion, optimizer, DEVICE, train_monitor)

        # Log training metrics to MLflow
        for key, value in train_metric_dict.items():
            mlflow.log_metric(f"train_{key}", value, step=epoch)

        # Step the scheduler
        scheduler.step()

        # Validate the model
        val_metric_dict = validate(model, val_loader, criterion, DEVICE, val_monitor)

        # Log validation metrics to MLflow
        for key, value in val_metric_dict.items():
            mlflow.log_metric(f"val_{key}", value, step=epoch)

        # Early stopping
        if val_monitor.early_stopping_check(val_metric_dict["dice_score"], model):
            break

    # Log the best model to MLflow after training
    print(f"Logging the best model with Dice Score of {val_monitor.best_score}")
    best_model_state = torch.load(val_monitor.export_path)
    model.load_state_dict(best_model_state)  # Load the best model state
    mlflow.pytorch.log_model(model, artifact_path="model")  # Log to MLflow
    
    #################### 2D Evaluation ####################
    predictions = []
    labels = []
    volume_idxs = []
    slice_idxs = []
    meta_data = val_dataset.metadata

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks, volume_idx, slice_idx) in enumerate(val_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            predictions.append(outputs)
            labels.append(masks)
            volume_idxs.extend(volume_idx.numpy().tolist())
            slice_idxs.extend(slice_idx.numpy().tolist())
            
    predictions = torch.cat(predictions, dim=0).detach().cpu()
    labels = torch.cat(labels, dim=0).squeeze(1).long().detach().cpu()
    dice_scores = dice_coefficient(predictions, labels, num_classes=4)
    print(dice_scores)
    print(f"Dice Score: {np.mean(dice_scores)}")
    
    #################### 3D Evaluation ####################
    N_TEST = 5
    metadata = val_dataset.metadata
    # Split the predictions and labels into volumes
    predictions = np.array(predictions).reshape(N_TEST, -1, 128, 256)
    labels = np.array(labels).reshape(N_TEST, -1, 128, 256)
    dices = dice_score_3d(predictions, labels, num_classes=4)
    print(dices)