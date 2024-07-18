
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class EventSpikeEncoder(nn.Module):
    def __init__(self):
        super(EventSpikeEncoder, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        return x

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 256, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.spike_encoder = EventSpikeEncoder()
        self.reduce_c1 = nn.Conv2d(256, 256, kernel_size=1)
        self.combine_features = nn.Conv2d(512, 256, kernel_size=1)
        self.corr_block = AttentionCorrelationBlock()
        self.update_block = UpdateBlock(args)
        self.context_network = ContextNetwork()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, event_volume):
        try:
            c1 = self.feature_encoder(event_volume)
            s1 = self.spike_encoder(event_volume)

            # デバッグプリントで形状を確認
            print(f"c1 shape: {c1.shape}")
            print(f"s1 shape: {s1.shape}")

            # c1のチャネル数をs1に合わせて減らす
            c1_reduced = self.reduce_c1(c1)

            # サイズを一致させるための処理を追加
            if c1_reduced.size() != s1.size():
                s1 = F.interpolate(s1, size=c1_reduced.shape[2:], mode='bilinear', align_corners=False)

            combined_features = torch.cat((c1_reduced, s1), dim=1)  # 特徴の結合
            combined_features = self.combine_features(combined_features)
            corr = self.corr_block(combined_features)
            flow_predictions = self.update_block(corr, event_volume)
            return flow_predictions
        except Exception as e:
            print(f"Error in RAFT forward pass: {e}")
            raise e

class AttentionCorrelationBlock(nn.Module):
    def __init__(self):
        super(AttentionCorrelationBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        try:
            b, c, h, w = x.size()
            x_flat = x.flatten(2).permute(2, 0, 1)
            attn_output, _ = self.attn(x_flat, x_flat, x_flat)
            attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)
            return attn_output
        except Exception as e:
            print(f"Error in AttentionCorrelationBlock forward pass: {e}")
            raise e

class UpdateBlock(nn.Module):
    def __init__(self, args):
        super(UpdateBlock, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, original_input):
        try:
            x1 = F.relu(self.conv1(x))
            x2 = F.relu(self.conv2(x1))
            attn_input = x2.flatten(2).permute(2, 0, 1)
            attn_output, _ = self.attn(attn_input, attn_input, attn_input)
            attn_output = attn_output.permute(1, 2, 0).view_as(x2)
            x2 = x2 + attn_output
            x3 = F.relu(self.conv3(x2))
            x4 = self.conv4(x3)
            flow = F.interpolate(x4, size=original_input.shape[2:], mode='bilinear', align_corners=False)
            return [flow]
        except Exception as e:
            print(f"Error in UpdateBlock forward pass: {e}")
            raise e

class ContextNetwork(nn.Module):
    def __init__(self):
        super(ContextNetwork, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(258, 128, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(128, 96, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=3, padding=16, dilation=16),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1, dilation=1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        try:
            return self.convs(x)
        except Exception as e:
            print(f"Error in ContextNetwork forward pass: {e}")
            raise e

class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SparseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        mask = (x != 0).float()
        out = self.conv(x * mask)
        return out * mask

class SparseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SparseResidualBlock, self).__init__()
        self.conv1 = SparseConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SparseConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                SparseConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class RAFTWithSparseConv(nn.Module):
    def __init__(self, args):
        super(RAFTWithSparseConv, self).__init__()
        self.feature_encoder = nn.Sequential(
            SparseConv2d(4, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            SparseResidualBlock(64, 128, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            SparseResidualBlock(128, 256, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            SparseResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.spike_encoder = EventSpikeEncoder()
        self.reduce_c1 = SparseConv2d(256, 256, kernel_size=1)
        self.combine_features = SparseConv2d(512, 256, kernel_size=1)
        self.corr_block = AttentionCorrelationBlock()
        self.update_block = UpdateBlock(args)
        self.context_network = ContextNetwork()

    def forward(self, event_volume):
        try:
            c1 = self.feature_encoder(event_volume)
            s1 = self.spike_encoder(event_volume)

            # デバッグプリントで形状を確認
            print(f"c1 shape: {c1.shape}")
            print(f"s1 shape: {s1.shape}")

            # c1のチャネル数をs1に合わせて減らす
            c1_reduced = self.reduce_c1(c1)

            # サイズを一致させるための処理を追加
            if c1_reduced.size() != s1.size():
                s1 = F.interpolate(s1, size=c1_reduced.shape[2:], mode='bilinear', align_corners=False)

            combined_features = torch.cat((c1_reduced, s1), dim=1)  # 特徴の結合
            combined_features = self.combine_features(combined_features)
            corr = self.corr_block(combined_features)
            flow_predictions = self.update_block(corr, event_volume)
            return flow_predictions
        except Exception as e:
            print(f"Error in RAFTWithSparseConv forward pass: {e}")
            raise e
