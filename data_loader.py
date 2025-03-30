import torch
import numpy as np
from torch.utils.data import DataLoader

class FluidDataset(torch.utils.data.Dataset):
    def __init__(self, density, velocity, inflow=None, obstacle=None, normalize=True):
        self.density = torch.tensor(density, dtype=torch.float32)
        #self.velocity = torch.tensor(velocity, dtype=torch.float32)
        self.velocity = torch.tensor(velocity, dtype=torch.float32)  # (C, H, W)
        self.inflow = torch.tensor(inflow, dtype=torch.float32) if inflow is not None else None
        self.obstacle = torch.tensor(obstacle, dtype=torch.float32) if obstacle is not None else None
        self.normalize = normalize
        self.time_steps = 2

        # Ensure all tensors have the same spatial size
        self.height, self.width = self.velocity.shape[-2:]  # Get H, W from velocity

        # Normalize data
        if normalize:
            self.density_min, self.density_max = self.density.min(), self.density.max()
            self.velocity_min, self.velocity_max = self.velocity.min(), self.velocity.max()
            self.density = (self.density - self.density.min()) / (self.density.max() - self.density.min() + 1e-8)
            self.velocity = (self.velocity - self.velocity.min()) / (self.velocity.max() - self.velocity.min() + 1e-8)
            print(self.velocity.shape)
            if self.inflow is not None:
                self.inflow = (self.inflow - self.inflow.min()) / (self.inflow.max() - self.inflow.min() + 1e-8)
            if self.obstacle is not None:
                self.obstacle = (self.obstacle - self.obstacle.min()) / (self.obstacle.max() - self.obstacle.min() + 1e-8)

    def __len__(self):
        return len(self.density) -self.time_steps
    
    def normalize_tensor(self, tensor, min_val, max_val):
        min_val = tensor.min()
        max_val = tensor.max()
        # if max_val - min_val < 1e-6:  # Prevent division by zero
        #     return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)
    
    def __getitem__(self, idx):
        # Extract last time step
        density_input = self.density[idx + self.time_steps - 1]  # (1, H, W)
        velocity_input = self.velocity[idx + self.time_steps - 1]  # (2, H, W)

        # Ensure correct shape: Remove batch dimension if needed
        if density_input.dim() == 3:  # (B, H, W) -> (H, W)
            density_input = density_input.squeeze(0)  # (H, W)

        density_input = density_input.unsqueeze(0)  # (1, H, W) to match velocity

        # Normalize if needed
        if self.normalize:
            density_input = self.normalize_tensor(density_input, self.density_min, self.density_max)
            velocity_input = self.normalize_tensor(velocity_input, self.velocity_min, self.velocity_max)

        # Stack input → (3, H, W) = [vx, vy, density]
        input_tensor = torch.cat([velocity_input, density_input], dim=0)  # (3, H, W)

        # Target: Next velocity field
        velocity_target = self.velocity[idx + self.time_steps]  # (2, H, W)

        return input_tensor, velocity_target

def load(data_file, batch_size=16):
    data = np.load(data_file)

    # Correct reshaping
    density = data['density'].squeeze(1).reshape(-1, 1, 128, 96)
    velocity = data['velocity'].squeeze(1)[..., :2].reshape(-1, 2, 128, 96)  # Keep only first 2 channels
    print(density.shape, velocity.shape)

    # Create dataset
    dataset = FluidDataset(density, velocity)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
    test_dataset = torch.utils.data.Subset(dataset, list(range(train_size, len(dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader