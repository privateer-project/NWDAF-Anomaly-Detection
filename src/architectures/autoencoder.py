import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
            self, input_size, hidden_size, output_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.output_size = output_size
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.lstm1.hidden_size,
            hidden_size=self.output_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, (hidden, _) = self.lstm2(x)
        x = self.layer_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout)
        self.lstm2 = nn.LSTM(input_size=self.lstm1.hidden_size,
                             hidden_size=self.output_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout,
                             )

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, dropout, **kwargs):
        super(LSTMAutoencoder, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = Encoder(input_size=self.input_size,
                               hidden_size=self.hidden_size1,
                               output_size=self.hidden_size2,
                               num_layers=self.num_layers,
                               dropout=self.dropout)
        self.decoder = Decoder(input_size=self.encoder.output_size,
                               hidden_size=self.hidden_size2,
                               output_size=self.hidden_size1,
                               num_layers=self.num_layers,
                               dropout=self.dropout)
        self.adaptive_average = nn.AdaptiveAvgPool2d((None, self.input_size))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.adaptive_average(decoded)
        return output

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded
