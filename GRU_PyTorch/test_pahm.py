import argparse
import matplotlib.pyplot as plt
import matplotlib
import torch
from dataloader import get_dataloaders
from pahm_model import PAHMModel
import numpy as np

def main():
    """
    To test a model, two parameters must be passed:
    --model_name: the name of the model file stored in train_pahm
    --extension: the same extension used by train_pahm when creating the model
    """

    parser = argparse.ArgumentParser(description='Test the PAHM Model.')
    parser.add_argument('--model_name', type=str, default='pahm_model.pth',
                        help='Path to the saved model file.')
    
    parser.add_argument('--extension', type=str,
                        default="none",
                        help='Feature extension (none,one,zero,past) (default "none": no extension.')
    parser.add_argument('--keep_angles', action='store_true',
                        help='Set this flag to not normalize angles.')
    parser.add_argument('--no_fig', action='store_true',
                        help='Set this flag to hide the figure.')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information.')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model name: '{args.model_name}'")

    
    model = PAHMModel.load_model(args.model_name)
    model = model.to(device)

    _, _, test_loader, _ = get_dataloaders(root_dir="../Datos_Recolectados/",extension=args.extension,normalize_angles=not args.keep_angles)


    predictions = []
    ground_truths = []
    pwm_values = []
    index = 0

    model.eval()
    with torch.no_grad():
        for pwm, lengths, angle in test_loader:
            # Initialize hidden state
            model.reset(batch_size=pwm.size(0))
            
            pwm = pwm.to(device)
            angle = angle.to(device)

            prediction = model(pwm,lengths)
            prediction = prediction.squeeze().cpu().numpy()
            
            predictions.append(prediction)
            ground_truths.append(angle.squeeze().cpu().numpy())
            pwm_values.append(pwm.squeeze().cpu().numpy())

            index += 1

            if args.verbose:
                print(f"Testing sequence {index}: ({prediction.shape[0]})")


    if model.use_hidden_offset and args.verbose:
        print("Offset for hidden state: ",model.hidden_offset.cpu().detach().numpy())
                
    show_results(pwm_values, predictions, ground_truths,hide_fig=args.no_fig)


def show_results(pwm_values,
                 predictions,
                 ground_truths,
                 prepadding=1500,
                 hide_fig=False):
    num_sequences = len(predictions)

    # Get a colormap with 20 colors
    colors = matplotlib.colormaps.get_cmap('tab20').colors  

    # Concatenate all predictions and ground truths
    all_predictions = np.concatenate(predictions)
    all_ground_truths = np.concatenate(ground_truths)

    # Calculate differences
    differences = all_predictions - all_ground_truths
    differences[0:prepadding]=0
   
    # Use numpy functions for MSE, MAE, and RMSE calculation
    mse = np.mean(differences**2)
    mae = np.mean(np.abs(differences))
    rmse = np.sqrt(mse)

    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)


    if not hide_fig:
        plt.figure(1,figsize=(15, 6))
        start_index=0
        for i in range(num_sequences):
            end_index = start_index + predictions[i].shape[0]
            plt.plot(
                range(start_index, end_index),
                all_predictions[start_index:end_index],
                color=colors[i % len(colors)],
                label="",
                linewidth=2
            )
            plt.plot(
                range(start_index, end_index),
                all_ground_truths[start_index:end_index],
                color=colors[i % len(colors)],
                linestyle="--",
                label=""
            )
            start_index=end_index

        plt.xlabel("Timesteps (concatenated sequences)")
        plt.ylabel("Angle")
        plt.title("Predictions and Ground Truth (concatenated)")

        plt.figure(2,figsize=(15, 6))
        start_index=0
        for i in range(num_sequences):
            end_index = start_index + predictions[i].shape[0]
            plt.plot(
                range(start_index, end_index),
                differences[start_index:end_index],
                color=colors[i],
                label="",
                linewidth=2
            )
            start_index=end_index

        plt.xlabel("Timesteps (concatenated sequences)")
        plt.ylabel("Difference (Prediction - Ground Truth)")
        plt.title("Differences between Predictions and Ground Truth (concatenated)")

        plt.show()


if __name__ == "__main__":
    main()
