import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

class PAHMModel(nn.Module):
    def __init__(self, input_size=1,
                 hidden_size=32,
                 output_size=1,
                 activation=nn.PReLU,
                 use_offset=False):
        """
        input_size: dimension of input vector: usually 1
        hidden_size: scalar or vector telling the size of the hidden state and 
                     subsequent layers
        output_size: size of the output (usually 1)
        activation: activation layer used after all layers
        offset:     if True, use a hidden offset
        """
        super(PAHMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        if not isinstance(hidden_size,list):
            hidden_size=[hidden_size]
        
        self.gru = nn.GRU(input_size, hidden_size[0], batch_first=True)

        # Create the hidden layers after the GRU
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_size)-1):
            self.hidden_layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.hidden_layers.append(self.activation())

        # Define the final output layer
        self.fc = nn.Linear(hidden_size[-1], output_size)
        self.hidden = None
        self.use_hidden_offset=use_offset

        if self.use_hidden_offset:
            self.hidden_offset = nn.Parameter(torch.zeros(hidden_size[0]),requires_grad=True)


    def forward(self, x, lengths=None):

        # Pack the sequences (i.e. tell gru to ignore padding)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths, batch_first=True)


        # Add the offset to the hidden state
        if self.use_hidden_offset and self.hidden is not None:
            self.hidden += self.hidden_offset
            
        # we need the hidden state for the next sequence
        out, self.hidden = self.gru(x, self.hidden)  

        # Subtract the offset from the new hidden state
        if self.use_hidden_offset:
            self.hidden -= self.hidden_offset
        
        # Unpack the output from GRU
        if isinstance(out, PackedSequence):
            out, _ = pad_packed_sequence(out, batch_first=True)

        for layer in self.hidden_layers:
            out = layer(out)
            
        out = self.fc(out)
        return out

    def predict(self, x):
        with torch.no_grad():
            x = x.unsqueeze(0)  # Add batch dimension
            out = self.forward(x)
            return out.squeeze(0)  # Remove batch dimension
    
    
    def reset(self,batch_size=1):
        """
        Resets the hidden state.  Used everytime the model needs to restart
        predictions
        """
        device = next(self.parameters()).device
        if isinstance(self.hidden_size,list):
            self.hidden = torch.zeros(1, batch_size, self.hidden_size[0]).to(device)
        else:
            self.hidden = torch.zeros(1, batch_size, self.hidden_size).to(device)

            
    def save_model(self, path):
        device = next(self.parameters()).device
        self.to('cpu')
        torch.save({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'activation': self.activation.__name__,
            'state_dict': self.state_dict(),
            'use_hidden_offset': self.use_hidden_offset
        }, path)
        self.to(device)

    @classmethod
    def load_model(cls, path,device=None):
        checkpoint = torch.load(path,map_location=device)
        if 'activation' in checkpoint:           
            activation = getattr(nn, checkpoint['activation'])  # Recreate the activation function
        else:
            activation = nn.PReLU

        if 'offset' in checkpoint:
            offset = checkpoint['offset']
        else:
            offset = False
            
        model = cls(checkpoint['input_size'],
                    checkpoint['hidden_size'],
                    checkpoint['output_size'],
                    activation,
                    use_offset=offset)
        model.load_state_dict(checkpoint['state_dict'])
        if device is not None:
            model.to(device)
        return model
