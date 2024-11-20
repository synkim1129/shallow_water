import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class WaterSurfaceUNet(nn.Module):
    def __init__(self, hidden_size=64, bilinear=True):
        super(WaterSurfaceUNet, self).__init__()
        self.hidden_size = hidden_size
        self.bilinear = bilinear

        # 입력: h_delta, u, v (3채널)
        self.inc = DoubleConv(3, hidden_size)
        self.down1 = Down(hidden_size, 2*hidden_size)
        self.down2 = Down(2*hidden_size, 4*hidden_size)
        self.down3 = Down(4*hidden_size, 8*hidden_size)
        factor = 2 if bilinear else 1
        self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
        self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
        self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
        self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
        self.up4 = Up(2*hidden_size, hidden_size, bilinear)
        # 출력: delta_h_delta, delta_u, delta_v (3채널)
        self.outc = OutConv(hidden_size, 3)

    def forward(self, h_delta, u, v):
        """
        입력:
            h_delta: 평균수심으로부터의 편차 (batch_size, 1, height, width)
            u: x방향 속도 (batch_size, 1, height, width)
            v: y방향 속도 (batch_size, 1, height, width)
        출력:
            h_delta_new: 다음 시점의 수심 편차
            u_new: 다음 시점의 x방향 속도
            v_new: 다음 시점의 y방향 속도
        """
        # 입력 텐서 결합
        x = torch.cat([h_delta, u, v], dim=1)
        
        # 인코더
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 디코더
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        # 변화량 예측
        delta_h_delta = x[:, 0:1]
        delta_u = x[:, 1:2]
        delta_v = x[:, 2:3]
        
        # 새로운 상태 계산
        h_delta_new = h_delta + delta_h_delta
        u_new = u + delta_u
        v_new = v + delta_v

        return h_delta_new, u_new, v_new