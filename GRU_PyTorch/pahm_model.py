import torch
import torch.nn as nn

class PAHMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PAHMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = None

    def forward(self, x):
        out, self.hidden = self.gru(x, self.hidden)  # we need the hidden state for the next sequence
        out = self.fc(out)
        return out

    def predict(self, x):
        with torch.no_grad():
            x = x.unsqueeze(0)  # Add batch dimension
            out = self.forward(x)
            return out.squeeze(0)  # Remove batch dimension
    
    
    def save_model(self, path):
        torch.save({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'state_dict': self.state_dict(),
        }, path)


    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path)
        model = cls(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['output_size'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
