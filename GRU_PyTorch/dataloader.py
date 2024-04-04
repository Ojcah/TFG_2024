import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PAHMDataset(Dataset):
  """
  A PyTorch Dataset class for loading PAHM data.
  """

  def __init__(self, root_dir, device, normalize=True):
    """
    Args:
      root_dir (str): Path to the directory containing CSV files.
      device (str): Device to store tensors ("cuda" or "cpu").
      normalize (bool, optional): Whether to normalize angle values. Defaults to True.
    """
    self.root_dir = root_dir
    self.device = device
    self.normalize = normalize
    self.data = []    

    # Read data from CSV files
    print("Loading data...")
    for filename in os.listdir(root_dir):
      data = pd.read_csv(os.path.join(root_dir, filename))
      pwm=np.concatenate((np.zeros(1500), data.values[:, 2]))
      angle=np.concatenate((np.zeros(1500), data.values[:, 3]))
      self.data.append((pwm,angle))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    # Preprocess data
    pwm, angle = self.data[idx]
    
    if self.normalize:
      angle = (angle - (-120)) / (120 - (-120))  # Normalize between -1 and 1

    # Convert to tensors and move to specified device
    pwm = torch.tensor(pwm, dtype=torch.float32).to(self.device)
    angle = torch.tensor(angle.reshape(-1, 1), dtype=torch.float32).to(self.device)

    return pwm, angle

def get_dataloaders(root_dir, train_split=0.6, val_split=0.2, device="cpu", normalize=True):
  """Creates training, validation, and testing dataloaders without randomization."""

  dataset = PAHMDataset(root_dir, device, normalize)
  total_len = len(dataset)

  train_len = int(train_split * total_len)
  val_len = int(val_split * total_len)
  test_len = total_len - train_len - val_len

  # Split the dataset preserving the order of time sequences
  train_data = dataset[:train_len]
  val_data = dataset[train_len: train_len + val_len]
  test_data = dataset[train_len + val_len:]

  train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)  # No shuffling for time sequences
  val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
  test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

  return train_dataloader, val_dataloader, test_dataloader

# Example usage
if __name__ == "__main__":
  root_dir = "../Datos_Recolectados/"
  device = "cuda" if torch.cuda.is_available() else "cpu"

  train_loader, val_loader, test_loader = get_dataloaders(root_dir, device=device)

  # Access data in batches
  for pwm, angle in train_loader:
    # Your training logic here
    pass
