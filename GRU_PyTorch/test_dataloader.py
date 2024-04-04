from dataloader import get_dataloaders
import os
import matplotlib.pyplot as plt
import torch

def plot_all_sequences(dataloaders):
  """
  Plots all sequences in the dataloaders with different colors.

  Args:
    dataloaders (tuple): A tuple containing training, validation, 
    and testing dataloaders.
  """
  colors = plt.cm.get_cmap('tab20').colors  # Get a colormap with 20 colors

  sequence_count = 0
  for loader in dataloaders:
    for pwm, angle in loader:
      # Plot PWM values
      plt.figure(1)
      plt.plot(pwm.cpu().numpy(),
               label=f"Sequence {sequence_count}",
               color=colors[sequence_count % len(colors)])

      # Plot angle values (separate figure)
      plt.figure(2)
      plt.plot(angle.cpu().numpy().squeeze(),
               label=f"Sequence {sequence_count}",
               color=colors[sequence_count % len(colors)])

      sequence_count += 1

  # Set labels and legends for PWM plot
  plt.figure(1)
  plt.xlabel("Index")
  plt.ylabel("PWM Value")
  plt.grid(True)
  plt.legend(title="Sequences")

  # Set labels and legends for angle plot
  plt.figure(2)
  plt.xlabel("Index")
  plt.ylabel("Angle (degrees)")
  plt.grid(True)
  plt.legend(title="Sequences")

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
 
  device = "cuda" if torch.cuda.is_available() else "cpu"

  train_loader, val_loader, test_loader = get_dataloaders(root_dir, device=device)

  plot_all_sequences((train_loader, val_loader, test_loader))
