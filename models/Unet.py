import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    A block consisting of two 3D convolutional layers, each followed by BatchNorm3d and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        """DoubleConv constructor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor.
        """
        
        return self.conv(x)

class UNet3D(nn.Module):
    """
    3D U-Net architecture for volumetric segmentation tasks.
    """
    def __init__(self, in_channels, out_channels, features=[32, 64, 128, 256]):
        """UNet3D constructor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            features (list, optional): Number of features in the encoder path. Defaults to [32, 64, 128, 256].
        """
        super(UNet3D, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.upsample.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Final Convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor.
        """
        # Encoder path
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for i in range(len(self.decoder)):
            x = self.upsample[i](x)
            skip_connection = skip_connections[i]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=True)
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i](x)

        return self.final_conv(x)
    
# Example usage
if __name__ == "__main__":
    # Hyperparameters
    in_channels = 1  # e.g., grayscale input (e.g., MRI)
    out_channels = 3  # e.g., 3 classes for segmentation
    features = [32, 64, 128, 256]

    # Model and input
    model = UNet3D(in_channels, out_channels, features)
    input_tensor = torch.randn(1, in_channels, 64, 128, 128)  # [Batch, Channels, Depth, Height, Width]

    # Forward pass
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Expected: [1, out_channels, 64, 128, 128]