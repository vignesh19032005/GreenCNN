"""
Example usage of GreenCNN for CIFAR-10 training
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from greencnn import GreenCNN, CarbonTracker, GreenTrainer


def load_cifar10_data(batch_size: int = 256):
    """Load CIFAR-10 dataset"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Test transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders (set num_workers=0 for Windows CUDA compatibility)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, test_loader


def main():
    """Main training example"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10_data(batch_size=256)
    
    # Create model optimized for speed and VRAM usage
    print("Creating GreenCNN model...")
    model = GreenCNN(
        num_classes=10,
        input_channels=3,
        base_channels=64,  # Balanced size for speed
        carbon_aware=True,
        adaptive_depth=True
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize carbon tracker
    carbon_tracker = CarbonTracker(
        project_name="GreenCNN_CIFAR10",
        country_iso_code="USA"
    )
    
    # Create trainer
    trainer = GreenTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        carbon_tracker=carbon_tracker,
        energy_budget=1.0,
        early_stopping_patience=15
    )
    
    # Training configuration
    num_epochs = 50
    save_path = "best_green_model.pth"
    
    print(f"🌱 Starting green training for {num_epochs} epochs...")
    
    # Train the model
    training_summary = trainer.train(
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    # Print final results
    print("\n" + "="*50)
    print("🌱 TRAINING COMPLETE!")
    print("="*50)
    
    print(f"Total Epochs: {training_summary['total_epochs']}")
    print(f"Best Validation Accuracy: {training_summary['best_val_accuracy']:.4f}")
    print(f"Final Validation Accuracy: {training_summary['final_val_accuracy']:.4f}")
    print(f"Best Carbon Efficiency: {training_summary['best_carbon_efficiency']:.2f} acc/g CO2")
    print(f"Average Carbon Efficiency: {training_summary['avg_carbon_efficiency']:.2f} acc/g CO2")
    
    carbon_summary = training_summary['carbon_summary']
    print(f"Total CO2 Emissions: {carbon_summary['total_emissions_kg']*1000:.2f}g")
    print(f"Total Energy Consumed: {carbon_summary['total_energy_kwh']:.4f} kWh")
    print(f"Training Duration: {carbon_summary['duration_seconds']/60:.1f} minutes")
    
    model_stats = training_summary['model_stats']
    print(f"Model Size: {model_stats['model_size_mb']:.2f} MB")
    print(f"Estimated FLOPs: {model_stats['estimated_flops']:,}")
    
    # Save carbon tracking data
    carbon_tracker.save_data("carbon_tracking_data.json")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    from greencnn.visualizer import visualize_training_results
    
    try:
        visualize_training_results(training_summary, "carbon_tracking_data.json")
        print("✅ Visualizations saved to current directory")
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
    
    print("\n🌱 Green training completed successfully!")


if __name__ == "__main__":
    main()