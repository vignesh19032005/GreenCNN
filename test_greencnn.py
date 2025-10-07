"""
Simple test script for GreenCNN components
"""

import torch
import numpy as np
from greencnn import GreenCNN, CarbonTracker, GreenOptimizer

def test_model():
    """Test GreenCNN model creation and forward pass"""
    print("Testing GreenCNN model...")
    
    model = GreenCNN(num_classes=10, carbon_aware=True, adaptive_depth=True)
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
    output = model(x)
    
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
    
    # Test model stats
    stats = model.get_model_stats()
    assert 'total_parameters' in stats
    assert 'estimated_flops' in stats
    
    print(f"✅ Model test passed! Parameters: {stats['total_parameters']:,}")

def test_carbon_tracker():
    """Test carbon tracking functionality"""
    print("Testing CarbonTracker...")
    
    tracker = CarbonTracker(project_name="test")
    
    # Start tracking
    tracker.start_tracking()
    
    # Simulate some work
    import time
    time.sleep(2)
    
    # Get current stats
    stats = tracker.get_current_stats()
    
    # Stop tracking
    summary = tracker.stop_tracking()
    
    assert 'total_emissions_kg' in summary
    assert summary['duration_seconds'] > 0
    
    print(f"✅ Carbon tracker test passed! Duration: {summary['duration_seconds']:.1f}s")

def test_green_optimizer():
    """Test green optimizer"""
    print("Testing GreenOptimizer...")
    
    model = GreenCNN(num_classes=10)
    optimizer = GreenOptimizer(model, base_optimizer="adam")
    
    # Simulate training step
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Calculate accuracy
    pred = output.argmax(dim=1)
    accuracy = (pred == y).float().mean().item()
    
    # Green optimization step
    optimizer.step(accuracy, 0.001)  # Small carbon emission
    
    stats = optimizer.get_stats()
    assert 'current_lr' in stats
    
    print(f"✅ Green optimizer test passed! LR: {stats['current_lr']:.6f}")

def test_adaptive_depth():
    """Test adaptive depth functionality"""
    print("Testing adaptive depth...")
    
    model = GreenCNN(num_classes=10, adaptive_depth=True)
    
    # Test energy budget adaptation
    model.adapt_to_energy_budget(0.5)  # 50% energy budget
    
    x = torch.randn(2, 3, 32, 32)
    output1 = model(x)
    
    # Increase energy budget
    model.adapt_to_energy_budget(1.0)  # 100% energy budget
    output2 = model(x)
    
    # Both should work
    assert output1.shape == output2.shape == (2, 10)
    
    print("✅ Adaptive depth test passed!")

def main():
    """Run all tests"""
    print("🧪 Running GreenCNN tests...\n")
    
    try:
        test_model()
        test_carbon_tracker()
        test_green_optimizer()
        test_adaptive_depth()
        
        print("\n🎉 All tests passed! GreenCNN is ready to use.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()