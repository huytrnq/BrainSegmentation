import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import BrainMRIDataset
from utils.utils import train, validate
from utils.vis import plot_mri

from models.Unet import UNet3D
from utils.loss import DiceCrossEntropyLoss

if __name__ == '__main__':
    ROOT_DIR = './Data/'
    BATCH_SIZE = 1
    EPOCHS = 50
    NUM_CLASSES = 4
    DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data Loader
    train_dataset = BrainMRIDataset(os.path.join(ROOT_DIR, 'train'), transform=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = BrainMRIDataset(os.path.join(ROOT_DIR, 'val'), transform=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES)
    model = model.to(DEVICE)
    
    # Loss
    criterion = DiceCrossEntropyLoss(is_3d=True)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training Loop
    # Example training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in progress_bar:
            images, masks = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)  # Adjust keys if necessary

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            #update the progress bar
            progress_bar.set_postfix("Loss: {:.4f}".format(epoch_loss / len(train_loader)))

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")