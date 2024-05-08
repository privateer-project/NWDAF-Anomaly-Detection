import os
import json
import pickle
import numpy as np
from datetime import datetime

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
    
def load_history_with_pickle(file_path):
    with open(file_path, 'rb') as file:
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
            
