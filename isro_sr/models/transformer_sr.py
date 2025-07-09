import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import List, Tuple, Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False,
                 drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(x, self.upscale_factor)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 64, num_blocks: int = 8):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.initial(x)
        features.append(x)
        
        for block in self.residual_blocks:
            x = block(x)
            features.append(x)
            
        return features

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x1, x2], dim=1)
        attention_weights = self.attention(combined)
        return x1 * attention_weights + x2 * (1 - attention_weights)

class SatelliteSR(nn.Module):
    def __init__(self, in_channels: int = 3, scale_factor: int = 4, base_channels: int = 64,
                 num_blocks: int = 8, num_heads: int = 8, transformer_layers: int = 4):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.feature_extractor1 = FeatureExtractor(in_channels, base_channels, num_blocks)
        self.feature_extractor2 = FeatureExtractor(in_channels, base_channels, num_blocks)
        
        # Enhanced multi-scale feature fusion
        self.fusion_layers = nn.ModuleList([
            AdaptiveFeatureFusion(base_channels) for _ in range(num_blocks + 1)
        ])
        
        # Multi-scale context aggregation
        self.context_aggregation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, dilation=1),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=2, dilation=2),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=4, dilation=4),
                nn.Conv2d(base_channels, base_channels, kernel_size=1)
            ) for _ in range(num_blocks + 1)
        ])
        
        # Transformer for global feature modeling with enhanced positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, (base_channels * 2) * (num_blocks + 1), base_channels))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(base_channels, num_heads, mlp_ratio=8.0) for _ in range(transformer_layers)
        ])
        
        # Enhanced upsampling with residual connections
        self.upconv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(int(math.log2(scale_factor)))
        ])
        
        # Multi-scale reconstruction
        self.multi_scale_recon = nn.ModuleList([
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(base_channels, in_channels, kernel_size=5, padding=2),
            nn.Conv2d(base_channels, in_channels, kernel_size=7, padding=3)
        ])
        
        # Final fusion for multi-scale outputs
        self.final_fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, base_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 16, base_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Extract multi-level features
        features1 = self.feature_extractor1(x1)
        features2 = self.feature_extractor2(x2)
        
        # Fuse features at each level with context aggregation
        fused_features = []
        for f1, f2, fusion_layer, context_layer in zip(features1, features2, self.fusion_layers, self.context_aggregation):
            fused = fusion_layer(f1, f2)
            # Apply multi-scale context aggregation
            context_enhanced = context_layer(fused)
            fused_features.append(context_enhanced)
        
        # Reshape for transformer
        B, C, H, W = fused_features[0].shape
        transformer_input = torch.cat([f.flatten(2).transpose(1, 2) for f in fused_features], dim=1)
        transformer_input = transformer_input + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            transformer_input = block(transformer_input)
        
        # Reshape back to spatial
        transformed = transformer_input.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply channel attention
        channel_att = self.channel_attention(transformed)
        transformed = transformed * channel_att
        
        # Apply spatial attention
        spatial_att_input = torch.cat([
            torch.mean(transformed, dim=1, keepdim=True),
            torch.max(transformed, dim=1, keepdim=True)[0]
        ], dim=1)
        spatial_att = self.spatial_attention(spatial_att_input)
        transformed = transformed * spatial_att
        
        # Enhanced progressive upsampling
        upsampled = transformed
        for upconv_layer in self.upconv_layers:
            upsampled = upconv_layer(upsampled)
        
        # Multi-scale reconstruction
        multi_scale_outputs = []
        for recon_layer in self.multi_scale_recon:
            multi_scale_outputs.append(recon_layer(upsampled))
        
        # Fuse multi-scale outputs
        multi_scale_concat = torch.cat(multi_scale_outputs, dim=1)
        output = self.final_fusion(multi_scale_concat)
        
        return output

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Load VGG for perceptual loss
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True).features[:35].eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
        except:
            self.vgg = None
            print("Warning: VGG19 not available for perceptual loss")
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor, quality_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Basic pixel losses
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        
        # Perceptual loss
        perceptual_loss = torch.tensor(0., device=pred.device)
        if self.vgg is not None:
            pred_features = self.vgg(pred)
            target_features = self.vgg(target)
            perceptual_loss = self.mse_loss(pred_features, target_features)
            
        # Edge loss using gradient
        edge_loss = self.gradient_loss(pred, target)
        
        # SSIM loss
        ssim_loss = 1 - self.ssim(pred, target)
        
        # Apply quality mask if provided
        if quality_mask is not None:
            quality_mask = quality_mask.expand_as(pred)
            l1 = (l1 * quality_mask).mean()
            mse = (mse * quality_mask).mean()
            perceptual_loss = (perceptual_loss * quality_mask).mean()
            edge_loss = (edge_loss * quality_mask).mean()
            ssim_loss = (ssim_loss * quality_mask).mean()
            
        # Combine losses
        total_loss = l1 + mse + 0.1 * perceptual_loss + 0.1 * edge_loss + 0.1 * ssim_loss
        return total_loss
    
    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def get_gradient(x):
            dx = x[..., :, 1:] - x[..., :, :-1]
            dy = x[..., 1:, :] - x[..., :-1, :]
            return dx, dy
        
        pred_dx, pred_dy = get_gradient(pred)
        target_dx, target_dy = get_gradient(target)
        
        grad_diff_x = self.mse_loss(pred_dx, target_dx)
        grad_diff_y = self.mse_loss(pred_dy, target_dy)
        
        return grad_diff_x + grad_diff_y
    
    def ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean() 