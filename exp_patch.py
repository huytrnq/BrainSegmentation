import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import dagshub
import mlflow
import mlflow.pytorch

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchio as tio
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchio")

from utils.utils import train_3d, validate_3d_patch, save_model_config_to_file
from utils.dataset import BrainMRIDataset
from utils.loss import DiceFocalLoss, DiceCrossEntropyLoss
from monai.networks.nets import UNet

if __name__ == '__main__':
    #################### Initialize DagsHub and MLflow ####################
    dagshub.init(repo_owner='huytrnq', repo_name='BrainSegmentation', mlflow=True)

    #################### Hyperparameters ####################
    ROOT_DIR = './Data'
    BATCH_SIZE = 1
    EPOCHS = 300
    NUM_CLASSES = 4
    NUM_WORKERS = 16
    DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    PATCH_SIZE = 128
    QUEUE_LENGTH = 200
    LR = 0.01
    MODEL_CONFIG_PATH = 'model_config.json'
    
    #################### DataLoaders ####################
    train_transform = tio.Compose([
        # tio.RandomElasticDeformation(num_control_points=(7, 7, 7), max_displacement=(4, 4, 4)),
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
    
    train_dataset = BrainMRIDataset(os.path.join(ROOT_DIR, 'train'), transform=train_transform)
    val_dataset = BrainMRIDataset(os.path.join(ROOT_DIR, 'val'), transform=val_transform)

    # sampler = tio.data.UniformSampler(PATCH_SIZE)
    # Create a LabelSampler that focuses on the specified label
    label_probabilities = {0: 1, 1: 3, 2: 2, 3: 2}

    sampler = tio.LabelSampler(
        patch_size=PATCH_SIZE,
        label_probabilities=label_probabilities,
    )
    # Create a queue to store patches
    patches_queue = tio.Queue(
        subjects_dataset=train_dataset,
        max_length=QUEUE_LENGTH,
        samples_per_volume=16,
        sampler=sampler,
        num_workers=0,
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    # Create a DataLoader that will iterate over patches
    train_loader = tio.SubjectsLoader(patches_queue, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    #################### Model ####################
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,  
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=4,
        norm="instance",
        dropout=0.1
    )
    model = model.to(DEVICE)
    save_model_config_to_file(model, MODEL_CONFIG_PATH)

    #################### Loss, Optimizer, and Scheduler ####################
    # criterion = DiceFocalLoss(alpha=0.5, gamma=2, is_3d=True, ignore_background=False)
    criterion = DiceCrossEntropyLoss(is_3d=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    #################### Start MLflow Run ####################
    mlflow.start_run(run_name="3D Brain MRI Segmentation - Patch")
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("num_classes", NUM_CLASSES)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("model", model.__class__.__name__)
    mlflow.log_param("criterion", criterion.__class__.__name__)
    # mlflow.log_param("alpha", criterion.alpha)
    # mlflow.log_param("gamma", criterion.gamma)
    # mlflow.log_param("focal_weight", criterion.focal_weight)
    # mlflow.log_param("dice_weight", criterion.dice_weight)
    mlflow.log_param("optimizer", optimizer.__class__.__name__)
    mlflow.log_param("scheduler", scheduler.__class__.__name__)
    mlflow.log_artifact(MODEL_CONFIG_PATH)

    #################### Training Loop ####################
    best_avg_dice = 0
    for epoch in range(EPOCHS):
        train_avg_loss, train_avg_dice, train_csf_dice, train_gm_dice, train_wm_dice = train_3d(
            model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS, NUM_CLASSES
        )
        val_loss, val_avg_dice, val_csf_dice, val_gm_dice, val_wm_dice = validate_3d_patch(
            model, val_dataset, criterion, epoch, EPOCHS, DEVICE, NUM_CLASSES, PATCH_SIZE, BATCH_SIZE, NUM_WORKERS
        )

        scheduler.step()

        # Log metrics to MLflow
        mlflow.log_metric("train_avg_loss", train_avg_loss, step=epoch)
        mlflow.log_metric("train_avg_dice", train_avg_dice, step=epoch)
        mlflow.log_metric("val_avg_loss", val_loss, step=epoch)
        mlflow.log_metric("val_avg_dice", val_avg_dice, step=epoch)
        mlflow.log_metric("val_csf_dice", val_csf_dice, step=epoch)
        mlflow.log_metric("val_gm_dice", val_gm_dice, step=epoch)
        mlflow.log_metric("val_wm_dice", val_wm_dice, step=epoch)

        if val_avg_dice > best_avg_dice:
            best_avg_dice = val_avg_dice
            torch.save(model.state_dict(), 'best_model_3d.pth')
            print(f'Best model saved with dice score: {best_avg_dice}\n')
    
    #################### Log Final Model ####################
    print(f'Best model achieved with dice score: {best_avg_dice}')
    mlflow.log_param("best_avg_dice", best_avg_dice)
    model.load_state_dict(torch.load('best_model_3d.pth'))
    mlflow.pytorch.log_model(model, artifact_path="model")
    mlflow.end_run()