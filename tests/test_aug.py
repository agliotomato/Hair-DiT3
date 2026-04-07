import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.augmentation import StrokeColorSampler

def test_stroke_color_sampler():
    print("Testing StrokeColorSampler...")
    
    # Create a 64x64 dummy data
    # Sketch with two "strokes": one red ([1, -1, -1]), one blue ([-1, -1, 1])
    sketch = torch.ones(3, 64, 64) * -1.0
    sketch[:, :32, :] = torch.tensor([1.0, -1.0, -1.0]).view(3, 1, 1) # Red top half
    sketch[:, 32:, :] = torch.tensor([-1.0, -1.0, 1.0]).view(3, 1, 1) # Blue bottom half
    
    # Target with a specific green color in the red sketch area
    # and a yellow color in the blue sketch area
    target = torch.ones(3, 64, 64) * -1.0
    target[:, :32, :] = torch.tensor([-1.0, 1.0, -1.0]).view(3, 1, 1) # Green
    target[:, 32:, :] = torch.tensor([1.0, 1.0, -1.0]).view(3, 1, 1)  # Yellow
    
    # Matte is 1 everywhere
    matte = torch.ones(1, 64, 64)
    
    sample = {
        "sketch": sketch.clone(),
        "target": target.clone(),
        "matte": matte.clone()
    }
    
    sampler = StrokeColorSampler(p=1.0)
    out_sample = sampler(sample)
    
    out_sketch = out_sample["sketch"]
    
    # Check if top half (originally red) is now green
    top_color = out_sketch[:, 0, 0]
    print(f"Original Red area color: {sketch[:, 0, 0].tolist()}")
    print(f"New sampled color: {top_color.tolist()}")
    
    # Check if bottom half (originally blue) is now yellow
    bottom_color = out_sketch[:, 33, 0]
    print(f"Original Blue area color: {sketch[:, 33, 0].tolist()}")
    print(f"New sampled color: {bottom_color.tolist()}")
    
    assert not torch.allclose(sketch, out_sketch), "Sketch should have changed!"
    print("Test passed!")

if __name__ == "__main__":
    test_stroke_color_sampler()
