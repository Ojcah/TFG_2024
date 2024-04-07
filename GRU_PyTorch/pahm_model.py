import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class PAHMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PAHMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = None

    def forward(self, x, lengths=None):

        # Pack the sequences (i.e. tell gru to ignore padding)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths, batch_first=True)
    
        out, self.hidden = self.gru(x, self.hidden)  # we need the hidden state for the next sequence

        # Unpack the output from GRU
        if isinstance(out, PackedSequence):
            out, _ = pad_packed_sequence(out, batch_first=True)
        
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
