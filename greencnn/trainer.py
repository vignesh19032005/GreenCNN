"""
Green Training: Carbon-aware training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import time
from tqdm import tqdm

from .model import GreenCNN
from .carbon_tracker import CarbonTracker, EpochCarbonTracker
from .green_optimizer import GreenOptimizer, EnergyAwareScheduler, GradientClippingGreen


class GreenTrainer:
    """
    Carbon-aware trainer for GreenCNN
    """
    
    def __init__(self, 
                 model: GreenCNN,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda",
                 carbon_tracker: Optional[CarbonTracker] = None,
                 energy_budget: float = 1.0,
                 early_stopping_patience: int = 10):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.energy_budget = energy_budget
        self.early_stopping_patience = early_stopping_patience
        
        # Carbon tracking
        self.carbon_tracker = carbon_tracker or CarbonTracker()
        self.epoch_tracker = EpochCarbonTracker(self.carbon_tracker)
        
        # Green optimization components
        self.green_optimizer = GreenOptimizer(
            model=model,
            base_optimizer="adamw",
            base_lr=0.001,
            carbon_tracker=self.carbon_tracker
        )
        
        self.scheduler = EnergyAwareScheduler(self.green_optimizer)
        self.gradient_clipper = GradientClippingGreen(max_norm=1.0, adaptive=True)
        
        # Training state
        self.training_history = []
        self.best_efficiency = 0
        self.best_accuracy = 0
        self.patience_counter = 0
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch with carbon tracking"""
        self.model.train()
        self.epoch_tracker.start_epoch(epoch)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.green_optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Green gradient clipping
            current_stats = self.carbon_tracker.get_current_stats()
            current_emissions = current_stats.get('total_emissions_kg', 0)
            self.gradient_clipper.clip_gradients(self.model, current_emissions)
            
            # Calculate accuracy for this batch
            pred = output.argmax(dim=1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            batch_accuracy = batch_correct / len(data)
            
            # Green optimization step
            self.green_optimizer.step(batch_accuracy, current_emissions)
            
            # Statistics
            running_loss += loss.item()
            correct += batch_correct
            total += len(data)
            
            # Update progress bar
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%',
                'CO2': f'{current_emissions*1000:.3f}g'
            })
            
            # Adapt model complexity based on energy budget
            if batch_idx % 100 == 0:  # Check every 100 batches
                energy_ratio = self._calculate_energy_ratio()
                self.model.adapt_to_energy_budget(energy_ratio)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct / total
        
        # End epoch tracking
        epoch_metrics = {
            'samples': total,
            'accuracy': epoch_accuracy,
            'loss': epoch_loss
        }
        
        epoch_carbon_data = self.epoch_tracker.end_epoch(epoch, epoch_metrics)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'carbon_data': epoch_carbon_data,
            'optimizer_stats': self.green_optimizer.get_stats()
        }
    
    def validate(self, epoch: int) -> Dict:
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        val_loss /= len(self.val_loader)
        val_accuracy = correct / total
        
        return {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
    
    def _calculate_energy_ratio(self) -> float:
        """Calculate current energy usage ratio"""
        current_stats = self.carbon_tracker.get_current_stats()
        current_power = current_stats.get('current_power_watts', 100)
        
        # Normalize against a baseline (e.g., 200W)
        baseline_power = 200
        energy_ratio = min(1.0, baseline_power / max(current_power, 50))
        
        return energy_ratio * self.energy_budget
    
    def train(self, num_epochs: int, 
              save_path: Optional[str] = None,
              log_callback: Optional[Callable] = None) -> Dict:
        """
        Main training loop with carbon tracking
        """
        
        print(f"🌱 Starting green training for {num_epochs} epochs")
        self.carbon_tracker.start_tracking()
        
        try:
            for epoch in range(num_epochs):
                # Training
                train_results = self.train_epoch(epoch)
                
                # Validation
                val_results = self.validate(epoch)
                
                # Combine results
                epoch_results = {**train_results, **val_results, 'epoch': epoch}
                self.training_history.append(epoch_results)
                
                # Calculate carbon efficiency
                carbon_data = train_results['carbon_data']
                if carbon_data:
                    current_efficiency = carbon_data['carbon_efficiency']
                    
                    # Update scheduler
                    lr_reduced = self.scheduler.step(current_efficiency)
                    
                    # Check for best efficiency
                    if current_efficiency > self.best_efficiency:
                        self.best_efficiency = current_efficiency
                        self.patience_counter = 0
                        
                        # Save best model
                        if save_path:
                            self.save_checkpoint(save_path, epoch, epoch_results)
                    else:
                        self.patience_counter += 1
                    
                    # Early stopping based on carbon efficiency
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch} due to carbon efficiency plateau")
                        break
                
                # Logging
                self._log_epoch_results(epoch, epoch_results)
                
                if log_callback:
                    log_callback(epoch_results)
        
        finally:
            # Stop carbon tracking
            final_summary = self.carbon_tracker.stop_tracking()
            
        # Training summary
        training_summary = self._create_training_summary(final_summary)
        
        return training_summary
    
    def _log_epoch_results(self, epoch: int, results: Dict):
        """Log epoch results"""
        carbon_data = results.get('carbon_data', {})
        optimizer_stats = results.get('optimizer_stats', {})
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {results['loss']:.4f}, Train Acc: {results['accuracy']:.4f}")
        print(f"  Val Loss: {results['val_loss']:.4f}, Val Acc: {results['val_accuracy']:.4f}")
        
        if carbon_data:
            print(f"  Carbon: {carbon_data['emissions_kg']*1000:.3f}g CO2")
            print(f"  Efficiency: {carbon_data['carbon_efficiency']:.2f} acc/g")
        
        if optimizer_stats:
            print(f"  Learning Rate: {optimizer_stats['current_lr']:.6f}")
            print(f"  LR Scale: {optimizer_stats['lr_scale_factor']:.3f}")
    
    def _create_training_summary(self, carbon_summary: Dict) -> Dict:
        """Create comprehensive training summary"""
        if not self.training_history:
            return {}
        
        # Extract metrics
        train_losses = [h['loss'] for h in self.training_history]
        train_accs = [h['accuracy'] for h in self.training_history]
        val_losses = [h['val_loss'] for h in self.training_history]
        val_accs = [h['val_accuracy'] for h in self.training_history]
        
        # Carbon efficiency data
        efficiencies = [h['carbon_data']['carbon_efficiency'] 
                       for h in self.training_history 
                       if h.get('carbon_data')]
        
        summary = {
            'total_epochs': len(self.training_history),
            'best_train_accuracy': max(train_accs),
            'best_val_accuracy': max(val_accs),
            'final_train_accuracy': train_accs[-1],
            'final_val_accuracy': val_accs[-1],
            'best_carbon_efficiency': max(efficiencies) if efficiencies else 0,
            'avg_carbon_efficiency': np.mean(efficiencies) if efficiencies else 0,
            'carbon_summary': carbon_summary,
            'model_stats': self.model.get_model_stats(),
            'efficiency_trend': self.epoch_tracker.get_efficiency_trend(),
            'training_history': self.training_history
        }
        
        return summary
    
    def save_checkpoint(self, path: str, epoch: int, results: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.green_optimizer.optimizer.state_dict(),
            'results': results,
            'training_history': self.training_history,
            'carbon_summary': self.carbon_tracker.get_current_stats()
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.green_optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint