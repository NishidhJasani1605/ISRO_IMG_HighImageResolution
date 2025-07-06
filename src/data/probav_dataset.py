import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
import h5py
import pandas as pd
from typing import Tuple, List, Optional
from .image_preprocessor import SatelliteImagePreprocessor
from .enhanced_preprocessor import EnhancedSatellitePreprocessor

class ProbaVDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 band: str = 'RED',
                 patch_size: int = 128,
                 scale_factor: int = 3,
                 augment: bool = True):
        """
        Dataset for PROBA-V super-resolution
        Args:
            root_dir: Root directory containing the data
            split: 'train' or 'test'
            band: 'RED' or 'NIR'
            patch_size: Size of training patches
            scale_factor: Super-resolution scale factor
            augment: Whether to apply augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.band = band
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment and split == 'train'
        
        # Load normalization parameters
        self.norm_params = pd.read_csv(self.root_dir / 'norm.csv')
        
        # Setup data paths
        self.data_dir = self.root_dir / split / band
        self.preprocessor = EnhancedSatellitePreprocessor()
        
        # Get all image sets
        self.image_sets = self._get_image_sets()
        
    def _get_image_sets(self) -> List[str]:
        """Get list of all image set IDs"""
        image_sets = []
        for f in self.data_dir.glob('*_LR.h5'):
            img_id = f.name.split('_')[0]
            if (self.data_dir / f'{img_id}_HR.h5').exists():
                image_sets.append(img_id)
        return image_sets
    
    def _load_h5_data(self, h5_path: Path) -> np.ndarray:
        """Load data from H5 file"""
        with h5py.File(h5_path, 'r') as f:
            data = f['data'][:]
        return data
    
    def _normalize_data(self, data: np.ndarray, img_id: str) -> np.ndarray:
        """Normalize data using parameters from norm.csv"""
        params = self.norm_params[self.norm_params['img_id'] == img_id].iloc[0]
        data = (data - params['mean']) / params['std']
        return data
    
    def _get_item_from_set(self, img_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get LR and HR data for a specific image set"""
        # Load LR and HR data
        lr_path = self.data_dir / f'{img_id}_LR.h5'
        hr_path = self.data_dir / f'{img_id}_HR.h5'
        
        lr_data = self._load_h5_data(lr_path)  # Shape: (T, H, W)
        hr_data = self._load_h5_data(hr_path)  # Shape: (H, W)
        
        # Normalize data
        lr_data = self._normalize_data(lr_data, img_id)
        hr_data = self._normalize_data(hr_data, img_id)
        
        # Select two best LR images
        if lr_data.shape[0] > 2:
            # TODO: Implement quality-based selection
            lr_data = lr_data[:2]
        
        # Register LR images using enhanced preprocessing
        lr1 = lr_data[0]
        lr2 = lr_data[1]
        registered_lr1, registered_lr2, quality_mask = self.preprocessor.enhance_image_pair(lr1, lr2)
        
        # Convert to torch tensors
        lr1_tensor = torch.from_numpy(registered_lr1).unsqueeze(0).float()
        lr2_tensor = torch.from_numpy(registered_lr2).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr_data).unsqueeze(0).float()
        quality_mask_tensor = torch.from_numpy(quality_mask).unsqueeze(0).float()
        
        return (lr1_tensor, lr2_tensor, quality_mask_tensor), hr_tensor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample"""
        img_id = self.image_sets[idx]
        return self._get_item_from_set(img_id)
    
    def __len__(self) -> int:
        return len(self.image_sets)

def create_probav_dataloaders(config: dict):
    """
    Create train and validation dataloaders for PROBA-V data
    Args:
        config: Configuration dictionary
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = ProbaVDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        band=config['data']['spectral_band'],
        patch_size=config['data']['patch_size'],
        scale_factor=config['data']['scale_factor'],
        augment=True
    )
    
    val_dataset = ProbaVDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        band=config['data']['spectral_band'],
        patch_size=config['data']['patch_size'],
        scale_factor=config['data']['scale_factor'],
        augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader 