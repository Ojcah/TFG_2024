import argparse
import matplotlib.pyplot as plt
import matplotlib
import torch
from dataloader import get_dataloaders
from pahm_model import PAHMModel
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Test the PAHM Model.')
    parser.add_argument('--model_name', type=str, default='pahm_model.pth',
                        help='Path to the saved model file.')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = get_dataloaders(root_dir="../Datos_Recolectados/", device=device)

    model = PAHMModel.load_model(args.model_name)
    model = model.to(device)

    predictions = []
    ground_truths = []
    pwm_values = []
    index = 0

    model.eval()
    with torch.no_grad():
        for pwm, angle in test_loader:
            pwm = pwm.to(device)
            model.hidden = None # Reset hidden state
            prediction = model.forward(pwm)
            prediction = prediction.squeeze().cpu().numpy()
            
            predictions.append(prediction)
            ground_truths.append(angle.squeeze().cpu().numpy())
            pwm_values.append(pwm.squeeze().cpu().numpy())

            index += 1
            print(f"Testing sequence {index}: ({prediction.shape[0]})")

    plot_results(pwm_values, predictions, ground_truths)

def plot_results(pwm_values, predictions, ground_truths):
    num_sequences = len(predictions)
    colors = matplotlib.colormaps.get_cmap('tab20').colors  # Get a colormap with 20 colors

    # Concatenate all predictions and ground truths
    all_predictions = np.concatenate(predictions)
    all_ground_truths = np.concatenate(ground_truths)

    
    plt.figure(1,figsize=(15, 6))
    for i in range(num_sequences):
        start_index = sum(predictions[j].shape[0] for j in range(i))  # Calculate starting index for each sequence
        end_index = start_index + predictions[i].shape[0]
        plt.plot(
            range(start_index, end_index),
            all_predictions[start_index:end_index],
            color=colors[i],
            label="",
            linewidth=2
        )
        plt.plot(
            range(start_index, end_index),
            all_ground_truths[start_index:end_index],
            color=colors[i],
            linestyle="--",
            label=""
        )

    plt.xlabel("Timesteps (concatenated sequences)")
    plt.ylabel("Angle")
    plt.title("Predictions and Ground Truth (concatenated)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)  # Hide legend

    # Calculate differences
    differences = all_predictions - all_ground_truths
    
    plt.figure(2,figsize=(15, 6))
    for i in range(num_sequences):
        start_index = sum(predictions[j].shape[0] for j in range(i))  # Calculate starting index for each sequence
        end_index = start_index + predictions[i].shape[0]
        plt.plot(
            range(start_index, end_index),
            differences[start_index:end_index],
            color=colors[i],
            label="",
            linewidth=2
        )
        
    plt.xlabel("Timesteps (concatenated sequences)")
    plt.ylabel("Difference (Prediction - Ground Truth)")
    plt.title("Differences between Predictions and Ground Truth (concatenated)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)  # Hide legend

    # Use numpy functions for MSE, MAE, and RMSE calculation
    mse = np.mean(differences**2)
    mae = np.mean(np.abs(differences))
    rmse = np.sqrt(mse)

    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)

    plt.show()

if __name__ == "__main__":
    main()
