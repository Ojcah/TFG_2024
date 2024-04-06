print("Loading packages...")
import argparse
import time
import datetime

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
                            default=500,
                            help='Number of training epochs.')
        parser.add_argument('--batch_size', type=int,
                            default=1,
                            help='Batch size used for training.')
        parser.add_argument('--model_name', type=str,
                            default='pahm_model',
                            help='Name for the RNAM model (.pth).')
        parser.add_argument('--loss_type', type=str,
                            default='mse',
                            help='Choose a loss type: mse or mae.')
        parser.add_argument('--load_model', type=str,
                            default='',
                            help='Load a previously trained model (.pth) and train it further.')
        parser.add_argument('--checkpoints_every', type=int,
                            default=0,
                            help='Number of epochs between checkpoints (default 0: save only at the end).')
        parser.add_argument('--extension', type=str,
                            default="none",
                            help='Feature extension (none,one,zero,past) (default "none": no extension.')
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
                "extension": self.args.extension
            })

    def train_model(self, model, train_loader, val_loader):
        # Define loss function and optimizer
        criterion = torch.nn.MSELoss() if self.args.loss_type == 'mse' else torch.nn.L1Loss()
        optimizer = torch.optim.NAdam(model.parameters())

        model = model.to(device)
        

        for epoch in range(self.args.epochs):
            model.train()            
            total_loss = 0
            for pwm, angle in train_loader:

                pwm = pwm.to(device)
                angle = angle.to(device)

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
                val_loss = sum(criterion(model(pwm.to(device)), \
                                               angle.to(device)) for pwm, angle in val_loader)

            print(f'Epoch {epoch+1}/{self.args.epochs}, '
                  f'Train Loss: {total_loss/len(train_loader)}, '
                  f'Validation Loss: {val_loss/len(val_loader)}')

            # Log losses to Weights & Biases
            if self.use_wandb:
                wandb.log({"Train Loss": total_loss/len(train_loader),
                           "Validation Loss": val_loss/len(val_loader)})
            
            # Save model checkpoints
            if (self.args.checkpoints_every > 0 and
                (epoch + 1) % self.args.checkpoints_every == 0):
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                num_digits = len(str(self.args.epochs))  # Get number of digits in epochs
                epoch_str = str(epoch + 1).zfill(num_digits)  # Pad epoch with zeros
                model_name = f"{self.args.wandb_run}_{epoch_str}_{timestamp}.pth"

                model.save_model(model_name)
                print(f"  Model saved at epoch {epoch + 1} to {model_name}")

        # Save model at the end of training
        model.save_model(self.args.model_name)
        print(f"Model saved at the end of training to {self.args.model_name}")
        return model

if __name__ == "__main__":
    trainer = Trainer()

    print("Cargando datos...")
    # Load data
    root_dir = "../Datos_Recolectados/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _, wholeset = get_dataloaders(root_dir,extension=trainer.args.extension)

    # Hyperparameters
    input_size = wholeset.features()  # Number of features in the input
    hidden_size = trainer.args.units  # Number of features in the hidden state
    output_size = 1  # Number of features in the output


    if trainer.args.load_model:
        model = PAHMModel.load_model(trainer.args.load_model)
    else:
        # Initialize model
        model = PAHMModel(input_size, hidden_size, output_size)

    print("Inicio de entrenamiento...")
        
    # Train model
    model = trainer.train_model(model, train_loader, val_loader)

    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{trainer.args.model_name}_{timestamp}.pth"

    model.save_model(model_name)

    wandb.finish()
