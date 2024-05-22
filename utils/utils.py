import os
import io
import json
import pickle
import torch
import numpy as np
from datetime import datetime

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def update_metadata_index(experiment_id, metadata, index_path='../results/experiments_metadata.json'):
    with open(index_path, 'r+') as file:
        index = json.load(file)
        index[experiment_id] = metadata
        file.seek(0)
        json.dump(index, file, indent=4)
        

def save_experiment_results(history, parameters, additional_metadata, experiment_id,  results_dir='../results'):
    
    history_file_path = os.path.join(results_dir, f"{experiment_id}_history.pkl")
    
    if os.path.exists(history_file_path):
        raise FileExistsError(f"File already exists: {history_file_path}")
    
    print(history_file_path)
    with open(history_file_path, 'wb') as file:
        pickle.dump(history, file)
    
    metadata = {
        'parameters': parameters,
        'min_train_loss': np.round(min(history.train_losses), 4),
        'min_val_loss': np.round(min(history.val_losses), 4),
        'min_train_val_gap': np.round(min([*map(lambda x: x[1] - x[0], zip(history.train_losses, history.val_losses))]), 4),
        'epochs_trained': history.epochs_trained,
        'results_file': history_file_path,
        'timestamp': str(datetime.now())
    }
    
    metadata.update(additional_metadata)
    
    update_metadata_index(experiment_id, metadata)
    
def load_history_from_pickle(file_path, device):
    with open(file_path, 'rb') as file:
        if device.type == 'cpu':
            history = CPU_Unpickler(file).load()
        else:
            history = pickle.load(file)
    return history

def print_parameters(parameters, experiment_id):
    print(f"Experiment {experiment_id} Parameters:\n")
    for key, value in parameters.items():
        if isinstance(value, bool):
            friendly_value = "Enabled" if value else "Disabled"
            print(f"{key.replace('_', ' ').capitalize()}: {friendly_value}")
        else:
            print(f"{key.replace('_', ' ').capitalize()}: {value}")
            
