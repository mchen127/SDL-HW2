import torch.nn as nn
import src.config as config

class LSTMForecast(nn.Module):
    """
    A simple LSTM-based forecasting model.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
            output_size (int): The number of output features.
        """
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.
        x shape: (batch_size, sequence_length, input_size)
        """
        # LSTM layer
        # out shape: (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x)

        # We only need the output from the last time step
        out = self.fc(out[:, -1, :])
        return out