import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PAHMDataset(Dataset):
  """
  A PyTorch Dataset class for loading PAHM data.

  This is device agnostic.

  You can normalize the output angles, but this expects the input data to be
  already normalized from 0 to 1.
  """

  def __init__(self,
               root_dir,
               normalize_angles=True,
               min_angle=-120,max_angle=120,
               extension=None):
    """
    Args:
      root_dir (str): Path to the directory containing CSV files.
      device (str): Device to store tensors ("cuda" or "cpu").
      normalize (bool, optional): Whether to normalize angle values. Defaults to True.
    """
    self.root_dir = root_dir
    self.normalize_angles = normalize_angles
    self.data = []
    self.min_angle = min_angle
    self.max_angle = max_angle

    self.num_features=[]


    # Read data from CSV files
    for filename in os.listdir(root_dir):
      data = pd.read_csv(os.path.join(root_dir, filename))
      pwm = np.concatenate((np.zeros(1500), data.values[:, 2]))

      # Modify pwm based on extension argument
      if extension == "zero":
        pwm = np.column_stack((pwm, np.zeros_like(pwm)))
        pwm[0,1]=0
        self.num_features=2
      elif extension == "one":
        pwm = np.column_stack((pwm, np.ones_like(pwm)))
        pwm[0,1]=0
        self.num_features=2
      elif extension == "past":
        # Shift elements, create a copy for the first element
        pwm = np.column_stack((pwm, np.roll(pwm, 1)))  
        pwm[0,1]=0
        self.num_features=2
      else:
        pwm = pwm.reshape(-1,1)
        self.num_features=1

      
      if self.normalize_angles:        
        angle=np.concatenate((np.zeros(1500), self.norm(data.values[:, 3])))
      else:
        angle=np.concatenate((np.zeros(1500), data.values[:, 3]))
        
      angle = angle.reshape(-1,1)

      pwm = torch.tensor(pwm, dtype=torch.float32)
      angle = torch.tensor(angle, dtype=torch.float32)

      self.data.append((pwm,angle))


  def norm(self,data):
      data = (data - self.min_angle) / (self.max_angle-self.min_angle)
      return data
                             
  def __len__(self):
    return len(self.data)
  
  def features(self):
    return self.num_features
    

  def __getitem__(self, idx):
    # Preprocess data
    pwm, angle = self.data[idx]
    return pwm, angle

def get_dataloaders(root_dir,
                    train_split=0.6,
                    val_split=0.2,
                    normalize_angles=True,
                    min_angle=-120,
                    max_angle=120,
                    extension=None):
  """Creates training, validation, and testing dataloaders without
     randomization."""

  dataset = PAHMDataset(root_dir=root_dir,normalize_angles=normalize_angles,
                        min_angle=min_angle,max_angle=max_angle,
                        extension=extension)
  total_len = len(dataset)

  train_len = int(train_split * total_len)
  val_len = int(val_split * total_len)
  
  # Split the dataset preserving the order of time sequences
  train_data = torch.utils.data.Subset(dataset, range(0, train_len))
  val_data = torch.utils.data.Subset(dataset, range(train_len, train_len + val_len))
  test_data = torch.utils.data.Subset(dataset, range(train_len + val_len, total_len))

  train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
  val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
  test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

  return train_dataloader, val_dataloader, test_dataloader, dataset

