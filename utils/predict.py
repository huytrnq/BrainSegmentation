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
    
    def predict_full_volume(self, dataloader, proba=False):
        """
        Perform full-volume prediction for all subjects in a dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
            proba (bool): Whether to return probabilities.

        Returns:
            list: List of predicted segmentation volumes or probabilities for each subject.
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["image"][tio.DATA].to(self.device)
                outputs = self.model(inputs)

                if proba:
                    predictions = torch.softmax(outputs, dim=1)  # Return probabilities
                else:
                    predictions = torch.argmax(outputs, dim=1)  # Return class labels

                results.append(predictions.cpu())
        
        return torch.stack(results, dim=0)

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
            prediction = torch.softmax(prediction, dim=0)  # probabilities
        else:
            prediction = torch.argmax(aggregator.get_output_tensor(), dim=0)  # Aggregate predictions
        return prediction.cpu()