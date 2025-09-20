import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size, seq_len):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # encoder
        self.encoder_lstm = nn.LSTM(input_size=n_features,
                                    hidden_size=hidden_size,
                                    batch_first=True)

        # decoder
        self.decoder_lstm = nn.LSTM(input_size=hidden_size,
                                    hidden_size=hidden_size,
                                    batch_first=True)
        self.decoder_out = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        repeated = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder_lstm(repeated)
        return self.decoder_out(decoded)