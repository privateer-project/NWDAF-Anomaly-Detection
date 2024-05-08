import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from training.training import TrainingHistory

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout, layer_norm_flag):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, num_layers=2, batch_first=True, dropout = dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, num_layers=2,  batch_first=True, dropout = dropout)
        self.layer_norm_flag = layer_norm_flag
        self.layer_norm = nn.LayerNorm(hidden_dim2)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, (hidden, _) = self.lstm2(x)
        if self.layer_norm_flag:
            x = self.layer_norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_dim, hidden_dim1, output_dim, dropout):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(encoded_dim, hidden_dim1, num_layers=2, batch_first=True, dropout = dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, output_dim, num_layers=2, batch_first=True, dropout = dropout)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x

class LSTMAutoencoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim1, 
        hidden_dim2, 
        output_dim, 
        dropout, 
        layer_norm_flag
    ):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, dropout, layer_norm_flag)
        self.decoder = Decoder(hidden_dim2, hidden_dim1, output_dim, dropout)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, x):
        decoded = self.decoder(x)
        return decoded
        
    def train_model(
        self, 
        num_epochs, 
        early_stopping, 
        train_data_loader, 
        val_data_loader, 
        mal_data_loader, 
        device, 
        criterion,  
        lr
    ):
        history = TrainingHistory()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.to(device)
        
        for epoch in range(num_epochs):
            self.train() 
            epoch_train_losses = []
            total_loss = 0
            
            progress_bar = tqdm(train_data_loader, desc='Training...')
            for inputs, targets in progress_bar:
                train_batch_size = inputs.shape[0]
                input_size = inputs.shape[-1]
                inputs, targets = inputs.view(train_batch_size,-1,input_size).to(device), targets.view(train_batch_size,-1,input_size).to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                for name, parameter in self.named_parameters():
                    if parameter.grad is not None:
                        grad_norm = parameter.grad.norm().item()
                        history.gradient_norms[name].append(grad_norm)
                
                optimizer.step()
                
                step_loss = loss.item()
                total_loss += step_loss
                epoch_train_losses.append(step_loss)
            
            avg_train_loss = total_loss / len(train_data_loader)
            history.train_losses.append(avg_train_loss)
            
            self.eval()
            total_val_loss = 0
            epoch_val_losses = []
            with torch.no_grad(): 
                progress_bar = tqdm(val_data_loader, desc='Validating...')
                for inputs, targets in progress_bar:
                    val_batch_size = inputs.shape[0]
                    input_size = inputs.shape[-1]
                    inputs, targets = inputs.view(val_batch_size,-1,input_size).to(device), targets.view(val_batch_size,-1,input_size).to(device)
                    outputs = self(inputs)
                    val_loss = criterion(outputs, targets)
                    step_val_loss = val_loss.item()
                    total_val_loss += step_val_loss
                    epoch_val_losses.append(step_val_loss)
            
            avg_val_loss = total_val_loss / len(val_data_loader)
            history.val_losses.append(avg_val_loss)
            
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            if early_stopping is not None:
                early_stopping(avg_val_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        
        self.eval()
        with torch.no_grad(): 
            for inputs, targets in mal_data_loader: 
                mal_batch_size = inputs.shape[0]
                input_size = inputs.shape[-1]
                inputs, targets = inputs.view(mal_batch_size,-1,input_size).to(device), targets.view(mal_batch_size,-1,input_size).to(device)
                outputs = self(inputs)
                mal_loss = criterion(outputs, targets)
                history.mal_losses.append(mal_loss)
                
        history.model_weights = self.state_dict()
        history.epochs_trained = epoch
        
        return history

