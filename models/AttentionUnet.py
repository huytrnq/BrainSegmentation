import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock3D, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv3d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AttentionUNet, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock3D(in_channels, 64)
        self.Conv2 = ConvBlock3D(64, 128)
        self.Conv3 = ConvBlock3D(128, 256)
        self.Conv4 = ConvBlock3D(256, 512)
        self.Conv5 = ConvBlock3D(512, 1024)

        self.Up5 = UpConv3D(1024, 512)
        self.Att5 = AttentionBlock3D(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock3D(1024, 512)

        self.Up4 = UpConv3D(512, 256)
        self.Att4 = AttentionBlock3D(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock3D(512, 256)

        self.Up3 = UpConv3D(256, 128)
        self.Att3 = AttentionBlock3D(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock3D(256, 128)

        self.Up2 = UpConv3D(128, 64)
        self.Att2 = AttentionBlock3D(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock3D(128, 64)

        self.Conv = nn.Conv3d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        return out