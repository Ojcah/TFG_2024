print("Loading packages...")
import argparse
import time
import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from dataloader import get_dataloaders
from pahm_model import PAHMModel

import wandb
import ast

class Trainer:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Mimetic Model for the Dynamical System.')
        parser.add_argument('--wandb_run', type=str,
                            default='',
                            help='Name of the run in Weights & Biases.')
        parser.add_argument('--hidden_size', type=str,
                            default='32',
                            help='Size of the hidden state. Use a single integer or lists [32 16]')
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
        parser.add_argument('--keep_angles', action='store_true',
                            help='Set this flag to not normalize angles.')
        parser.add_argument('--offset', action='store_true',
                            help='Set this flag to use an internal offset for the hidden state.')
        self.args = parser.parse_args()

        # Parse hidden_size
        try:
            self.args.hidden_size = ast.literal_eval(self.args.hidden_size)
        except ValueError:
            print("Invalid format for --hidden_size. It should be an integer or a list of integers.")
            exit(1)

        # Check if hidden_size is a list of integers
        if isinstance(self.args.hidden_size, list):
            if not all(isinstance(i, int) for i in self.args.hidden_size):
                print("--hidden_size list should only contain integers.")
                exit(1)
        # Check if hidden_size is an integer
        elif not isinstance(self.args.hidden_size, int):
            print("--hidden_size should be an integer or a list of integers.")
            exit(1)
        
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
                "hidden_size": self.args.hidden_size,
                "learning_rate": 0.001,
                "extension": self.args.extension,
                "normalize_angles": not self.args.keep_angles
            })

    def create_mask(self,lengths, max_length,prepadding=1500):
        mask = torch.arange(max_length).expand(len(lengths),
                                               max_length) < lengths.unsqueeze(1)

        # This doesn't affect training, but testing => reset of hidden
        # state is done for every seq.  mask[:, :prepadding] = 0 # Set
        # the first 'prepadding' samples to zero
        return mask.float()
            
    def train_model(self, model, train_loader, val_loader):
        # Define loss function and optimizer
        criterion = torch.nn.MSELoss() if self.args.loss_type == 'mse' else torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())

        model = model.to(device)        

        for epoch in range(self.args.epochs):
            model.train()            
            total_loss = 0
            for pwm, lengths, angle in train_loader:

                # Initialize hidden state
                model.reset(batch_size=pwm.size(0))
                
                pwm = pwm.to(device)
                angle = angle.to(device)
                   

                # Create a mask for the current batch
                mask = self.create_mask(lengths, pwm.size(1)).to(device)
                
                outputs = model(pwm,lengths)

                # Apply the mask to the outputs and angle
                masked_outputs = outputs * mask.unsqueeze(-1)
                masked_angle = angle * mask.unsqueeze(-1)
                
                loss = criterion(masked_outputs, masked_angle)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for pwm, lengths, angle in val_loader:
                    # Initialize hidden state
                    model.reset(batch_size=pwm.size(0))

                    pwm = pwm.to(device)
                    angle = angle.to(device)

                    # Create a mask for the current batch
                    mask = self.create_mask(lengths, pwm.size(1)).to(device)
                    
                    outputs = model(pwm,lengths)

                    # Apply the mask to the outputs and angle
                    masked_outputs = outputs * mask.unsqueeze(-1)
                    masked_angle = angle * mask.unsqueeze(-1)
                    
                    loss = criterion(masked_outputs, masked_angle)
                    val_loss += loss.item()

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

        return model

if __name__ == "__main__":
    trainer = Trainer()

    print("Cargando datos...")
    # Load data
    root_dir = "../Datos_Recolectados/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _, wholeset = get_dataloaders(root_dir,
                                                            extension=trainer.args.extension,
                                                            normalize_angles=not trainer.args.keep_angles,
                                                            batch_size=trainer.args.batch_size)

    # Hyperparameters
    input_size = wholeset.features()  # Number of features in the input
    hidden_size = trainer.args.hidden_size  # Dimension of the hidden state
    output_size = 1  # Number of features in the output
    use_offset = trainer.args.offset

    print(f"Input size  : {input_size}")

    if isinstance(trainer.args.hidden_size, int):       
        print(f"Hidden state: {hidden_size}")
    else:
        print(f"Hidden state: {hidden_size[0]}")
    
        print("  Other layers: ",', '.join(map(str,hidden_size[1:])))

    print(f"Output size : {output_size}")
    print(f"Hidden offset: {use_offset}")

    
    if trainer.args.load_model:
        model = PAHMModel.load_model(trainer.args.load_model,device=device)
    else:
        # Initialize model
        model = PAHMModel(input_size, hidden_size, output_size, use_offset=use_offset)

    print("Inicio de entrenamiento...")
        
    # Train model
    model = trainer.train_model(model, train_loader, val_loader)

    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{trainer.args.model_name}_{timestamp}.pth"

    model.save_model(model_name)
    print(f"Model saved at the end of training to {model_name}")

    wandb.finish()
