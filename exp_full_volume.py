import os
import dagshub
dagshub.init(repo_owner='huytrnq', repo_name='BrainSegmentation', mlflow=True)

import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchio as tio
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchio")

from utils.utils import train_3d, validate_3d, save_model_config_to_file
from utils.dataset import BrainMRIDataset
from utils.loss import DiceCrossEntropyLoss, DiceFocalLoss
from monai.networks.nets import UNet

if __name__ == '__main__':
    #################### Hyperparameters ####################
    ROOT_DIR = './Data'
    BATCH_SIZE = 1
    PATEINCE = 50
    EPOCHS = 300
    NUM_CLASSES = 4
    NUM_WORKERS = 16
    DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 0.01
    MODEL_CONFIG_PATH = 'model_config.json'

    #################### DataLoaders ####################
    # TorchIO transformations for augmentation
    train_transform = tio.Compose([
        # tio.RandomElasticDeformation(num_control_points=7, locked_borders=2),
        tio.RandomFlip(axes=(0, 1, 2)),
        tio.RandomBiasField(coefficients=(0.1, 0.5), order=3),
        tio.RandomGamma(log_gamma=(-0.3, 0.3)),
        
        tio.RescaleIntensity((0, 1)),
        tio.ZNormalization(),
    ])

    val_transform = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.ZNormalization(),
    ])

    # Load the dataset
    train_dataset = BrainMRIDataset(os.path.join(ROOT_DIR, 'train'), transform=train_transform)
    val_dataset = BrainMRIDataset(os.path.join(ROOT_DIR, 'val'), transform=val_transform)

    # Create DataLoaders
    train_loader = tio.SubjectsLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = tio.SubjectsLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    #################### Model ####################
    model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4,  
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=4,
            norm="instance",  # Use Instance Normalization
            dropout=0.1       # Add dropout
        )

    model = model.to(DEVICE)
    save_model_config_to_file(model, MODEL_CONFIG_PATH)

    #################### Loss, Optimizer, Scheduler ####################
    # criterion = DiceFocalLoss(alpha=[0.05, 0.5, 0.25, 0.35], gamma=2, is_3d=True, ignore_background=False, focal_weight=0.5, dice_weight=0.5)
    criterion = DiceCrossEntropyLoss(is_3d=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-9)
    
    #################### Start MLflow Run ####################
    mlflow.start_run(run_name="3D Brain MRI Segmentation")
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("num_classes", NUM_CLASSES)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("model", model.__class__.__name__)
    # Loss
    mlflow.log_param("criterion", criterion.__class__.__name__)
    # mlflow.log_param("alpha", criterion.alpha)
    # mlflow.log_param("gamma", criterion.gamma)
    # mlflow.log_param("focal_weight", criterion.focal_weight)
    # mlflow.log_param("dice_weight", criterion.dice_weight)
    # Transformer
    mlflow.log_param("train_transform", train_transform.transforms)
    mlflow.log_param("val_transform", val_transform.transforms)
    
    mlflow.log_param("optimizer", optimizer.__class__.__name__)
    mlflow.log_param("scheduler", scheduler.__class__.__name__)
    mlflow.log_artifact(MODEL_CONFIG_PATH)

    #################### Training Loop ####################
    model_export_path = f"model_{model.__class__.__name__}_3d.pth"
    best_avg_dice = 0
    best_epoch = 0
    for epoch in range(EPOCHS):
        train_avg_loss, train_avg_dice, train_csf_dice, train_gm_dice, train_wm_dice = train_3d(model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS, NUM_CLASSES)
        val_avg_loss, val_avg_dice, val_csf_dice, val_gm_dice, val_wm_dice = validate_3d(model, val_loader, criterion, DEVICE, epoch, EPOCHS, NUM_CLASSES)

        scheduler.step()

        # Log metrics to MLflow
        mlflow.log_metric("train_avg_loss", train_avg_loss, step=epoch)
        mlflow.log_metric("train_avg_dice", train_avg_dice, step=epoch)
        mlflow.log_metric("val_avg_loss", val_avg_loss, step=epoch)
        mlflow.log_metric("val_avg_dice", val_avg_dice, step=epoch)
        mlflow.log_metric("val_csf_dice", val_csf_dice, step=epoch)
        mlflow.log_metric("val_gm_dice", val_gm_dice, step=epoch)
        mlflow.log_metric("val_wm_dice", val_wm_dice, step=epoch)
        
        # Patience
        if epoch - best_epoch > PATEINCE:
            print(f'Early stopping at epoch {epoch}')
            break
        
        if val_avg_dice > best_avg_dice:
            best_avg_dice = val_avg_dice
            best_epoch = epoch
            torch.save(model.state_dict(), model_export_path)
            print(f'Best model saved with dice score: {best_avg_dice}\n')
        

    #################### Log Final Model ####################
    print(f'Best model achieved at epoch {best_epoch} with dice score: {best_avg_dice}')
    mlflow.log_param("best_avg_dice", best_avg_dice)
    model.load_state_dict(torch.load(model_export_path))
    mlflow.pytorch.log_model(model, artifact_path="model")
    mlflow.end_run()