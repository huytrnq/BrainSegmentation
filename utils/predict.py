import torch
import torchio as tio
import numpy as np
import mlflow.pytorch

class Predictor:
    def __init__(self, model=None, device='cpu', patch_size=None, mlflow_model_uri=None):
        """
        Initialize the Predictor class.

        Args:
            model (torch.nn.Module, optional): The trained model. Not required if using MLflow to load the model.
            device (str): Device to perform inference ('cuda', 'cpu', etc.).
            patch_size (int, optional): Patch size for patch-based inference.
            mlflow_model_uri (str, optional): URI of the MLflow model to load.
        """
        self.device = device
        self.patch_size = patch_size
        
        if mlflow_model_uri:
            self.model = mlflow.pytorch.load_model(mlflow_model_uri).to(device)
        elif model:
            self.model = model.to(device)
        else:
            raise ValueError("Either 'model' or 'mlflow_model_uri' must be provided.")
    
    def predict_full_volume(self, volume, batch_size=1, proba=False):
        """
        Perform full-volume prediction.

        Args:
            volume (torch.Tensor): Input volume with shape [1, D, H, W].
            batch_size (int): Batch size for prediction.
            proba (bool): Whether to return probabilities.

        Returns:
            torch.Tensor: Predicted segmentation volume.
        """
        self.model.eval()
        volume = volume.to(self.device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = self.model(volume)
        if proba:
            predictions = outputs
        else:
            predictions = torch.argmax(outputs, dim=1).squeeze(0)  # Remove batch dimension
        return predictions.cpu()

    def predict_patches(self, subject, batch_size=1, overlap=0, proba=False):
        """
        Perform patch-based inference with reconstruction.

        Args:
            subject (tio.Subject): Input subject containing the volume.
            batch_size (int): Batch size for prediction.
            overlap (int): Overlap between patches.
            proba (bool): Whether to return probabilities.

        Returns:
            torch.Tensor: Reconstructed segmentation volume.
        """
        assert self.patch_size is not None, "Patch size must be defined for patch-based prediction."
        grid_sampler = tio.inference.GridSampler(subject, patch_size=self.patch_size, patch_overlap=overlap)
        aggregator = tio.inference.GridAggregator(grid_sampler)

        self.model.eval()
        with torch.no_grad():
            for patches_batch in torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size):
                inputs = patches_batch["image"][tio.DATA].to(self.device)
                outputs = self.model(inputs)
                locations = patches_batch[tio.LOCATION]
                aggregator.add_batch(outputs, locations)

        if proba:
            prediction = aggregator.get_output_tensor()
        else:
            prediction = torch.argmax(aggregator.get_output_tensor(), dim=0)  # Aggregate predictions
        return prediction.cpu()