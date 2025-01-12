import os
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
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

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='huytrnq', repo_name='BrainSegmentation', mlflow=True)

# Hyperparameters
ROOT_DIR = './Data'
BATCH_SIZE = 1
EPOCHS = 250
NUM_CLASSES = 4
NUM_WORKERS = 8
DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
PATCH_SIZE = 128
QUEUE_LENGTH = 200
LR = 0.01
MODEL_CONFIG_PATH = 'model_config.json'
KFOLD = 5
PATIENCE = 40

# Data Loaders
train_transform = tio.Compose([
    tio.RescaleIntensity((0, 1)),
    tio.ZNormalization(),
])
val_transform = tio.Compose([
    tio.RescaleIntensity((0, 1)),
    tio.ZNormalization(),
])

# Full Dataset
full_dataset = BrainMRIDataset(os.path.join(ROOT_DIR, 'train'), transform=train_transform)

# K-Fold Cross-Validation
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
fold_results = []

# Start MLflow Run
# mlflow.start_run(run_name="3D Brain MRI Segmentation - Cross-Validation")
mlflow.start_run(run_id='47240a9f9b9248e089e3ccefc97616d6')
mlflow.log_param("kfold", KFOLD)
mlflow.log_param("epochs", EPOCHS)
mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("learning_rate", LR)
mlflow.log_param("patch_size", PATCH_SIZE)

# Cross-Validation Loop
for fold, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):
    print(f"Starting Fold {fold + 1}/{KFOLD}")
    if fold < 2:
        continue
    if fold != 2 or fold != 3:
        mlflow.log_param(f"fold_{fold + 1}", {"train_indices": train_indices.tolist(), "val_indices": val_indices.tolist()})

    # Split Dataset
    train_subjects = [full_dataset[i] for i in train_indices]
    val_subjects = [full_dataset[i] for i in val_indices]

    train_dataset = tio.SubjectsDataset(train_subjects)
    val_dataset = tio.SubjectsDataset(val_subjects)

    # Sampling Strategy
    label_probabilities = {0: 1, 1: 3, 2: 2, 3: 2}
    sampler = tio.LabelSampler(patch_size=PATCH_SIZE, label_probabilities=label_probabilities)

    # Queue for Training
    patches_queue = tio.Queue(
        subjects_dataset=train_dataset,
        max_length=QUEUE_LENGTH,
        samples_per_volume=16,
        sampler=sampler,
        num_workers=0,
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    train_loader = tio.SubjectsLoader(patches_queue, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=4,
        norm="instance",
        dropout=0.1,
    )
    model = model.to(DEVICE)
    save_model_config_to_file(model, MODEL_CONFIG_PATH)

    # Loss, Optimizer, Scheduler
    criterion = DiceCrossEntropyLoss(is_3d=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # Training Loop
    best_avg_dice = 0
    patience = 0
    for epoch in range(EPOCHS):
        train_avg_loss, train_avg_dice, train_csf_dice, train_gm_dice, train_wm_dice = train_3d(
            model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS, NUM_CLASSES
        )
        val_loss, val_avg_dice, val_csf_dice, val_gm_dice, val_wm_dice = validate_3d_patch(
            model, val_dataset, criterion, epoch, EPOCHS, DEVICE, NUM_CLASSES, PATCH_SIZE, BATCH_SIZE, NUM_WORKERS
        )
        scheduler.step()

        # Log Metrics
        mlflow.log_metric(f"fold_{fold + 1}_train_avg_loss", train_avg_loss, step=epoch)
        mlflow.log_metric(f"fold_{fold + 1}_train_avg_dice", train_avg_dice, step=epoch)
        mlflow.log_metric(f"fold_{fold + 1}_val_avg_loss", val_loss, step=epoch)
        mlflow.log_metric(f"fold_{fold + 1}_val_avg_dice", val_avg_dice, step=epoch)
        
        # Early Stopping
        if val_avg_dice > best_avg_dice:
            patience = 0
            best_avg_dice = val_avg_dice
            torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')
            print(f"Saving Best Model for Fold {fold + 1} at Epoch {epoch + 1} with Dice Score: {best_avg_dice}\n")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early Stopping for Fold {fold + 1} at Epoch {epoch + 1} with Dice Score: {best_avg_dice}\n")
                break

    fold_results.append(best_avg_dice)
    print(f"Best Dice Score for Fold {fold + 1}: {best_avg_dice}")
    mlflow.log_param(f"fold_{fold + 1}_best_avg_dice", best_avg_dice)
    mlflow.pytorch.log_model(model, f"models/fold_{fold + 1}")

# Final Ensemble Validation
print("Performing Final Ensemble Validation...")
ensemble_dice_scores = []
for val_subject in val_dataset:
    subject_dices = []
    for fold in range(KFOLD):
        model.load_state_dict(torch.load(f'best_model_fold_{fold + 1}.pth'))
        dice_score = validate_3d_patch(
            model, [val_subject], criterion, 0, 1, DEVICE, NUM_CLASSES, PATCH_SIZE, BATCH_SIZE, NUM_WORKERS
        )[1]
        subject_dices.append(dice_score)
    ensemble_dice_scores.append(np.mean(subject_dices))

final_dice_score = np.mean(ensemble_dice_scores)
print(f"Final Ensemble Dice Score: {final_dice_score}")
mlflow.log_metric("final_ensemble_dice_score", final_dice_score)

# End MLflow Run
mlflow.end_run()