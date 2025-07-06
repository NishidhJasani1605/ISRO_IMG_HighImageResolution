import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import logging
from typing import Optional, Dict, List, Tuple
import wandb
from tqdm import tqdm
import yaml
import random
from dotenv import load_dotenv
from torch.optim.swa_utils import AveragedModel, SWALR
import kornia.augmentation as K
from kornia.metrics import ssim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from models.transformer_sr import SatelliteSR, CustomLoss
from data.probav_dataset import create_probav_dataloaders

# Load environment variables
load_dotenv()

class AdvancedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Load VGG for perceptual loss (if available)
        try:
            import torchvision.models as models
            vgg = models.vgg19(weights='IMAGENET1K_V1').features[:35].eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg.cuda() if torch.cuda.is_available() else vgg
        except:
            self.vgg = None
            print("Warning: VGG19 not available for perceptual loss")
    
    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss"""
        return 1.0 - ssim(pred.unsqueeze(1), 
                         target.unsqueeze(1), 
                         window_size=11, 
                         max_val=1.0,
                         reduction='mean')
        
    def spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss using FFT"""
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        return self.l1_loss(pred_mag, target_mag)
    
    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge-preserving loss"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if pred.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
        
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-8)
        
        return self.l1_loss(pred_edge, target_edge)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, quality_mask: torch.Tensor) -> torch.Tensor:
        # Basic losses
        l1 = self.l1_loss(pred * quality_mask, target * quality_mask)
        mse = self.mse_loss(pred * quality_mask, target * quality_mask)
        
        # SSIM loss
        ssim_loss = self.ssim_loss(pred, target)
        
        # Perceptual loss
        perceptual_loss = torch.tensor(0., device=pred.device)
        if self.vgg is not None and pred.shape[1] == 1:
            # Convert single channel to 3 channels for VGG
            pred_3ch = pred.repeat(1, 3, 1, 1)
            target_3ch = target.repeat(1, 3, 1, 1)
            
            try:
                pred_features = self.vgg(pred_3ch)
                target_features = self.vgg(target_3ch)
                perceptual_loss = self.mse_loss(pred_features, target_features)
            except:
                perceptual_loss = torch.tensor(0., device=pred.device)
        
        # Edge loss
        edge_loss_val = self.edge_loss(pred, target)
        
        # Spectral loss
        spectral_loss_val = self.spectral_loss(pred, target)
        
        # Combine losses with optimized weights
        total_loss = (
            0.3 * l1 + 
            0.25 * mse + 
            0.2 * ssim_loss + 
            0.1 * perceptual_loss + 
            0.1 * edge_loss_val + 
            0.05 * spectral_loss_val
        )
        
        return total_loss

class AdvancedTrainer:
    def __init__(self, config: Dict):
        """
        Initialize the trainer with configuration
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_distributed()
        
        # Initialize models, optimizers, and schedulers
        self.models = []
        self.optimizers = []
        self.schedulers = []
        
        # Create model ensemble
        for _ in range(3):  # Using 3 models for ensemble
            model = SatelliteSR(
                in_channels=config['model']['in_channels'],
                scale_factor=config['data']['scale_factor'],
                base_channels=config['model']['base_channels'],
                num_blocks=config['model']['num_blocks'],
                num_heads=config['model']['num_heads'],
                transformer_layers=config['model']['transformer_layers']
            )
            if self.distributed:
                model = DDP(model, device_ids=[self.local_rank])
            else:
                model = model.to(self.device)
            
            self.models.append(model)
            
            # Parse learning rate and weight decay as float
            lr = float(config['training']['learning_rate'])
            weight_decay = float(config['training']['weight_decay'])
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
            
            # Use cosine annealing with warm restarts if specified
            if config['training'].get('cosine_annealing', False):
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=config['training'].get('T_max', 100),
                    eta_min=float(config['training'].get('eta_min', 1e-7))
                )
            else:
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=lr,
                    epochs=config['training']['num_epochs'],
                    steps_per_epoch=config['training']['steps_per_epoch'],
                    pct_start=0.3,
                    anneal_strategy='cos'
                )
            
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
        
        # Initialize SWA model
        self.swa_model = AveragedModel(self.models[0])
        self.swa_scheduler = SWALR(
            self.optimizers[0],
            swa_lr=float(config['training'].get('swa_lr', 1e-6))
        )
        
        # Advanced loss function
        self.criterion = AdvancedLoss().to(self.device)
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        
        # Advanced augmentations
        self.augmentations = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=90),
            K.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            K.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0))
        ).to(self.device)
        
        # Initialize logging
        if self.is_main_process:
            self.setup_logging()
            if not config.get('disable_wandb', False):
                self.setup_wandb()
            
        # Curriculum learning parameters
        self.curriculum_step = 0
        self.difficulty_factors = np.linspace(0.5, 1.0, config['training']['num_epochs'])
        
    def setup_distributed(self):
        """Setup distributed training if available"""
        self.distributed = False
        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1
            
        if self.distributed:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
        else:
            self.local_rank = 0
            self.world_size = 1
            
        self.is_main_process = self.local_rank == 0
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['checkpoint_dir'], exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config['logging']['log_dir']}/training.log"),
                logging.StreamHandler()
            ]
        )
        
    def setup_wandb(self):
        """Initialize Weights & Biases logging with API key from environment"""
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.config['logging']['run_name'],
                config=self.config
            )
        else:
            logging.warning("WANDB_API_KEY not found in environment variables. Skipping wandb initialization.")
            self.config['disable_wandb'] = True
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Enhanced training for one epoch with curriculum learning and ensemble"""
        for model in self.models:
            model.train()
            
        total_loss = 0
        total_psnr = 0
        num_batches = len(dataloader)
        
        # Set curriculum difficulty
        difficulty = self.difficulty_factors[min(epoch, len(self.difficulty_factors)-1)]
        
        with tqdm(total=num_batches, disable=not self.is_main_process) as pbar:
            for batch_idx, ((lr1, lr2, quality_mask), hr) in enumerate(dataloader):
                lr1, lr2 = lr1.to(self.device), lr2.to(self.device)
                quality_mask = quality_mask.to(self.device)
                hr = hr.to(self.device)
                
                # Apply advanced augmentations
                if random.random() < difficulty:  # Curriculum learning
                    lr1 = self.augmentations(lr1)
                    lr2 = self.augmentations(lr2)
                    hr = self.augmentations(hr)
                
                # Forward pass with mixed precision for each model
                ensemble_outputs = []
                ensemble_losses = []
                
                for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                    with autocast():
                        sr_output = model(lr1, lr2)
                        loss = self.criterion(sr_output, hr, quality_mask)
                        ensemble_outputs.append(sr_output)
                        ensemble_losses.append(loss)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    
                    # Update learning rate
                    self.schedulers[i].step()
                
                # Average predictions and losses
                sr_output = torch.stack(ensemble_outputs).mean(0)
                loss = torch.stack(ensemble_losses).mean()
                
                # Update SWA model
                if epoch >= self.config['training']['swa_start_epoch']:
                    self.swa_model.update_parameters(self.models[0])
                    self.swa_scheduler.step()
                
                # Calculate metrics
                with torch.no_grad():
                    psnr = -10 * torch.log10(torch.mean((sr_output - hr) ** 2))
                    total_loss += loss.item()
                    total_psnr += psnr.item()
                
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item(), 'psnr': psnr.item()})
                
                if self.is_main_process and batch_idx % self.config['training']['log_interval'] == 0:
                    wandb.log({
                        'train_batch_loss': loss.item(),
                        'train_batch_psnr': psnr.item(),
                        'learning_rate': self.schedulers[0].get_last_lr()[0],
                        'difficulty': difficulty
                    })
                    
        return total_loss / num_batches, total_psnr / num_batches
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Enhanced validation with ensemble predictions and test-time augmentation"""
        for model in self.models:
            model.eval()
        self.swa_model.eval()
        
        total_loss = 0
        total_psnr = 0
        num_batches = len(dataloader)
        
        for (lr1, lr2, quality_mask), hr in tqdm(dataloader, disable=not self.is_main_process):
            lr1, lr2 = lr1.to(self.device), lr2.to(self.device)
            quality_mask = quality_mask.to(self.device)
            hr = hr.to(self.device)
            
            # Test-time augmentation with ensemble predictions
            sr_outputs = []
            
            # Get predictions from each model
            for model in self.models + [self.swa_model]:
                model_outputs = []
                for flip_h in [True, False]:
                    for flip_v in [True, False]:
                        for rot in [0, 90, 180, 270]:
                            lr1_aug = self.apply_augmentation(lr1, flip_h, flip_v, rot)
                            lr2_aug = self.apply_augmentation(lr2, flip_h, flip_v, rot)
                            sr_aug = model(lr1_aug, lr2_aug)
                            sr_aug = self.reverse_augmentation(sr_aug, flip_h, flip_v, rot)
                            model_outputs.append(sr_aug)
                
                # Average predictions for this model
                sr_outputs.append(torch.stack(model_outputs).mean(0))
            
            # Average predictions across all models
            sr_output = torch.stack(sr_outputs).mean(0)
            
            # Calculate metrics
            loss = self.criterion(sr_output, hr, quality_mask)
            psnr = -10 * torch.log10(torch.mean((sr_output - hr) ** 2))
            
            total_loss += loss.item()
            total_psnr += psnr.item()
            
        return total_loss / num_batches, total_psnr / num_batches
    
    def apply_augmentation(self, x: torch.Tensor, flip_h: bool, flip_v: bool, rot: int) -> torch.Tensor:
        """Apply augmentation to input tensor"""
        if flip_h:
            x = torch.flip(x, [-1])
        if flip_v:
            x = torch.flip(x, [-2])
        if rot:
            x = torch.rot90(x, k=rot//90, dims=[-2, -1])
        return x
    
    def reverse_augmentation(self, x: torch.Tensor, flip_h: bool, flip_v: bool, rot: int) -> torch.Tensor:
        """Reverse augmentation"""
        if rot:
            x = torch.rot90(x, k=4-(rot//90), dims=[-2, -1])
        if flip_v:
            x = torch.flip(x, [-2])
        if flip_h:
            x = torch.flip(x, [-1])
        return x
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Enhanced checkpoint saving with ensemble models"""
        if not self.is_main_process:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dicts': [model.state_dict() for model in self.models],
            'swa_model_state_dict': self.swa_model.state_dict(),
            'optimizer_state_dicts': [opt.state_dict() for opt in self.optimizers],
            'scheduler_state_dicts': [sch.state_dict() for sch in self.schedulers],
            'metrics': metrics
        }
        
        if (epoch + 1) % self.config['training']['checkpoint_interval'] == 0:
            checkpoint_path = f"{self.config['logging']['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")
        
        if is_best:
            best_path = f"{self.config['logging']['checkpoint_dir']}/best_model.pth"
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model with metrics: {metrics}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Main training loop
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        best_psnr = float('-inf')
        
        for epoch in range(self.config['training']['num_epochs']):
            if self.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            # Train for one epoch
            train_loss, train_psnr = self.train_epoch(train_loader, epoch)
            
            # Validate
            metrics = {}
            if val_loader is not None:
                val_loss, val_psnr = self.validate(val_loader)
                metrics.update({
                    'val_loss': val_loss,
                    'val_psnr': val_psnr
                })
                
                # Save best model
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    self.save_checkpoint(epoch, metrics, is_best=True)
            
            # Log metrics
            if self.is_main_process:
                metrics.update({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_psnr': train_psnr
                })
                wandb.log(metrics)
                
            # Regular checkpoint saving
            self.save_checkpoint(epoch, metrics)
                
        if self.is_main_process:
            wandb.finish()

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Add advanced training parameters for maximum accuracy
    config['training'].update({
        'num_models': 5,  # Increased ensemble size
        'swa_start_epoch': int(0.8 * config['training']['num_epochs']),  # Start SWA later
        'swa_lr': 5e-6,  # Lower SWA learning rate
        'steps_per_epoch': None,  # Will be set after dataloader creation
        'batch_size': config['data']['batch_size'],  # Ensure consistency
        'num_workers': config['data']['num_workers']
    })
    
    # Create trainer
    trainer = AdvancedTrainer(config)
    
    # Create dataloaders
    train_loader, val_loader = create_probav_dataloaders(config)
    config['training']['steps_per_epoch'] = len(train_loader)
    
    logging.info(f"Created dataloaders. Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}")
    
    # Start training
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main() 