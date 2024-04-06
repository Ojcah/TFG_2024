print("Loading packages...")

from dataloader import get_dataloaders
import os
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np

def plot_all_sequences(dataloaders):
  """
  Plots all sequences in the dataloaders with different colors.

  Args:
    dataloaders (tuple): A tuple containing training, validation, 
    and testing dataloaders.
  """
  # Get a colormap with 20 colors
  colors = matplotlib.colormaps.get_cmap('tab20').colors 

  fig_base = 1
  for loader in dataloaders: 
    # We'll plot a pair of figures for each dataloader
    print(f"Plotting set {(fig_base+1)/2}")
    pwms = []
    angles = []

    # First, pack all sequences in a lilst
    for pwm, angle in loader:
      pwm = pwm.squeeze().cpu().numpy()[:,0]
      angle = angle.squeeze().cpu().numpy()
      
      pwms.append(pwm)
      angles.append(angle)

    # Now crunch all of them as a single, long one, together.
    num_sequences=len(pwms)
    all_pwms = np.concatenate(pwms)
    all_angles = np.concatenate(angles)

    start_index=0
    for i in range(num_sequences):
      # Plot now each subsequence in its own place
      # First the PWM vallues
      end_index = start_index + pwms[i].shape[0]
      plt.figure(fig_base)
      plt.plot(
        range(start_index, end_index),
        all_pwms[start_index:end_index],
        color=colors[i % len(colors)],
        label="",
        linewidth=2
      )
      # And now the angles
      plt.figure(fig_base+1)
      plt.plot(
        range(start_index, end_index),
        all_angles[start_index:end_index],
        color=colors[i % len(colors)],
        label="",
        linewidth=2
      )
      start_index=end_index

    # Set labels and legends for PWM plot
    plt.figure(fig_base)
    plt.xlabel("Index")
    plt.ylabel(f"PWM Value {(fig_base+1)/2}")
    plt.grid(True)

    # Set labels and legends for angle plot
    plt.figure(fig_base+1)
    plt.xlabel("Index")
    plt.ylabel(f"Angle (degrees) {(fig_base+1)/2}")
    plt.grid(True)

    fig_base = fig_base + 2

  # Show both plots
  plt.show()

# Example usage
if __name__ == "__main__":
  root_dir = "../Datos_Recolectados/"

  if not os.path.exists(root_dir):
    print(f"Error: No existe el directorio '{root_dir}'.")
    print("Descargue los datos de Jorge Brenes en \n"+
          "  k80:/home/jbrenes/home/datos_jorge.tar.bz2\n"+
          "y descomprima en el padre de este directorio.")
    exit()  # Exit the script if the directory is missing
 
  print("Loading data...")
  train_loader, val_loader, test_loader, _ = get_dataloaders(root_dir,extension="zero")

  plot_all_sequences((train_loader, val_loader, test_loader))
