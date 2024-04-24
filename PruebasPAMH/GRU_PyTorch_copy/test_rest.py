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

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PAHMModel.load_model(args.model_name)
    model = model.to(device)

    predictions = []
    ground_truths = []
    pwm_values = []
    index = 0

    model.eval()

    length=5000
    
    with torch.no_grad():
        # Initialize hidden state
        model.reset()
            
        # Create a numpy array to hold the predictions for this sequence
        prediction = np.zeros(length)

        for i in range(length):  
            pwm_sample = torch.zeros(1,1).to(device)
            angle = model.predict(pwm_sample)
            prediction[i] = angle.cpu().numpy().item()


    print("Hidden state:",model.hidden)
          
    plt.plot(range(length),
             prediction,
             label="",
             linewidth=2
             )

    plt.xlabel("Timesteps")
    plt.ylabel("Predicted angle")
    plt.title("Rest response")

    plt.show()

if __name__ == "__main__":
    main()
