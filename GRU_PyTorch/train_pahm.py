print("Loading packages...")
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import get_dataloaders
from pahm_model import PAHMModel

import wandb

class Trainer:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Mimetic Model for the Dynamical System.')
        parser.add_argument('--wandb_run', type=str,
                            default='',
                            help='Name of the run in Weights & Biases.')
        parser.add_argument('--units', type=int,
                            default=32,
                            help='Number of units for the RNAM.')
        parser.add_argument('--epochs', type=int,
                            default=1000,
                            help='Number of training epochs.')
        parser.add_argument('--batch_size', type=int,
                            default=1,
                            help='Batch size used for training.')
        parser.add_argument('--loss_name', type=str,
                            default='loss_',
                            help='Name for the loss figure (.png).')
        parser.add_argument('--predict_name', type=str,
                            default='Prediction_',
                            help='Name for the prediction figure (.png).')
        parser.add_argument('--model_name', type=str,
                            default='Model_Synth_',
                            help='Name for the RNAM model (.pth).')
        parser.add_argument('--loss_type', type=str,
                            default='mse',
                            help='Choose a loss type: mse or mae.')
        parser.add_argument('--load_model', type=str,
                            default='',
                            help='Load a previously trained model (.pth) and train it further.')
        self.args = parser.parse_args()


        self.use_wandb = bool(self.args.wandb_run)

        if self.use_wandb:
            # Remember to follow the instructions at
            #   https://wandb.ai/mimetic-rna/pytorch-gru
            # before running this or it won't succeed.       
            wandb.login()
            wandb.init(project="pytorch-gru", 
                       entity="mimetic-rna", 
                       name=self.args.wandb_run,
                       resume='Allow',
                       id=self.args.wandb_run)
            
            wandb.config.update({
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                "units": self.args.units,
                "learning_rate": 0.001,
            })

    def train_model(self, model, train_loader, val_loader):
        # Define loss function and optimizer
        criterion = torch.nn.MSELoss() if self.args.loss_type == 'mse' else torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())

        model = model.to(device)
        

        for epoch in range(self.args.epochs):
            model.train()            
            total_loss = 0
            for pwm, angle in train_loader:

                # Forward pass
                outputs = model(pwm)
                loss = criterion(outputs, angle)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.hidden = None # Reset hidden state
                total_loss += loss.item()
                
            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = sum(criterion(model(pwm.to(device)), angle.to(device)) for pwm, angle in val_loader)

            print(f'Epoch {epoch+1}/{self.args.epochs}, Train Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

            # Log losses to Weights & Biases
            if self.use_wandb:
                wandb.log({"Train Loss": total_loss/len(train_loader), "Validation Loss": val_loss/len(val_loader)})
            
        return model

if __name__ == "__main__":
    trainer = Trainer()

    # Hyperparameters
    input_size = 1  # Number of features in the input
    hidden_size = trainer.args.units  # Number of features in the hidden state
    output_size = 1  # Number of features in the output

    print("Cargando datos...")
    # Load data
    root_dir = "../Datos_Recolectados/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _ = get_dataloaders(root_dir, device=device)

    if trainer.args.load_model:
        model = PAHMModel.load_model(trainer.args.load_model)
    else:
        # Initialize model
        model = PAHMModel(input_size, hidden_size, output_size)

    print("Inicio de entrenamiento...")
        
    # Train model
    model = trainer.train_model(model, train_loader, val_loader)

    # Save model
    model.save_model(trainer.args.model_name)
