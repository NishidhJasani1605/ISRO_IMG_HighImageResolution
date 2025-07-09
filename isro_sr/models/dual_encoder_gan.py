import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class DualEncoderBlock(nn.Module):
    """
    Dual encoder block for processing two input images
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DualEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x1, x2):
        # Process first image
        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        
        # Process second image
        out2 = self.conv2(x2)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        
        return out1, out2

class AttentionFusion(nn.Module):
    """
    Attention-based fusion of features from two encoders
    """
    def __init__(self, channels):
        super(AttentionFusion, self).__init__()
        self.conv_att1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_att2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_out = nn.Conv2d(channels*2, channels, kernel_size=1)
    
    def forward(self, x1, x2):
        # Attention weights
        att1 = torch.sigmoid(self.conv_att1(x1))
        att2 = torch.sigmoid(self.conv_att2(x2))
        
        # Apply attention
        feat1 = x1 * att1
        feat2 = x2 * att2
        
        # Concatenate and fuse features
        fused = self.conv_out(torch.cat([feat1, feat2], dim=1))
        
        return fused

class DecoderBlock(nn.Module):
    """
    Decoder block for super-resolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class UpscaleBlock(nn.Module):
    """
    Upscaling block for super-resolution
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpscaleBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                         kernel_size=scale_factor, 
                                         stride=scale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        out = self.upconv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class DualEncoderGANSR(nn.Module):
    """
    Dual-Encoder GAN for Super-Resolution
    """
    def __init__(self, in_channels=3, out_channels=3, hidden_dim=64, n_blocks=4, scale_factor=2):
        super(DualEncoderGANSR, self).__init__()
        
        # Dual encoder blocks
        self.encoder_blocks = nn.ModuleList([
            DualEncoderBlock(in_channels if i == 0 else hidden_dim * (2**min(i, 3)), 
                            hidden_dim * (2**min(i+1, 3)))
            for i in range(n_blocks)
        ])
        
        # Attention fusion blocks
        self.fusion_blocks = nn.ModuleList([
            AttentionFusion(hidden_dim * (2**min(i+1, 3)))
            for i in range(n_blocks)
        ])
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(hidden_dim * (2**min(n_blocks-i, 3)), 
                        hidden_dim * (2**min(n_blocks-i-1, 3)))
            for i in range(n_blocks-1)
        ])
        
        # Upscale blocks
        self.upscale = UpscaleBlock(hidden_dim, hidden_dim, scale_factor)
        
        # Final output layer
        self.output_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x1, x2):
        # Encoder
        encoder_features1 = []
        encoder_features2 = []
        
        for block in self.encoder_blocks:
            x1, x2 = block(x1, x2)
            encoder_features1.append(x1)
            encoder_features2.append(x2)
        
        # Fusion
        fused_features = []
        for i, fusion in enumerate(self.fusion_blocks):
            fused = fusion(encoder_features1[i], encoder_features2[i])
            fused_features.append(fused)
        
        # Decoder with skip connections
        x = fused_features[-1]
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            x = x + fused_features[-(i+2)]  # Skip connection
        
        # Upscale
        x = self.upscale(x)
        
        # Output
        x = self.output_conv(x)
        x = torch.tanh(x)  # Output in range [-1, 1]
        
        return (x + 1) / 2  # Convert to [0, 1] range
    
    def generate_sr(self, img1, img2, augment=False):
        """
        Generate super-resolution image from two low-resolution inputs
        
        Args:
            img1: First input image (numpy array or torch tensor)
            img2: Second input image (numpy array or torch tensor)
            augment: Whether to use mixed-pixel augmentation
            
        Returns:
            Super-resolution output image
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(img1, np.ndarray):
            img1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        if isinstance(img2, np.ndarray):
            img2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Get device
        device = next(self.parameters()).device
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        # Apply mixed-pixel augmentation if enabled
        if augment:
            # Simple mixed-pixel augmentation (more sophisticated methods would be used in practice)
            mixed = 0.7 * img1 + 0.3 * img2
            img1 = mixed
        
        # Generate super-resolution output
        with torch.no_grad():
            output = self.forward(img1, img2)
        
        return output
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Loaded model
        """
        # Create a new model instance
        model = cls()
        
        # Load the state dict
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Load the state dict
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model 