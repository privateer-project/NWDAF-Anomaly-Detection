import torch
from tqdm import tqdm
import numpy as np

from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    roc_curve,
    auc,
    recall_score
)


def evaluate(model, criterion, benign_test_data_loader, mal_test_data_loader, device):
    
    model.eval()
    benign_test_losses = []
    with torch.no_grad(): 
        progress_bar = tqdm(benign_test_data_loader, desc='Validating...')
        for inputs, targets in progress_bar:
            test_batch_size = inputs.shape[0]
            test_output_dim = inputs.shape[-1]
            inputs, targets = inputs.view(test_batch_size,-1,test_output_dim).to(device), targets.view(test_batch_size,-1,test_output_dim).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            benign_test_losses.append(loss.item())
            
    model.eval()
    mal_test_losses = []
    with torch.no_grad(): 
        progress_bar = tqdm(mal_test_data_loader, desc='Validating...')
        for inputs, targets in progress_bar:
            test_batch_size = inputs.shape[0]
            test_output_dim = inputs.shape[-1]
            inputs, targets = inputs.view(test_batch_size,-1,test_output_dim).to(device), targets.view(test_batch_size,-1,test_output_dim).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mal_test_losses.append(loss.item())
            
    return benign_test_losses, mal_test_losses


def calculate_threshold(benign_test_losses, mal_test_losses):
    
    fpr, tpr, thresholds = roc_curve(
        [0] * len(benign_test_losses) + [1] * len(mal_test_losses), 
        benign_test_losses + mal_test_losses
    )
    roc_auc = auc(fpr, tpr)
        
    distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)

    min_distance_idx = np.argmin(distances)

    optimal_threshold = thresholds[min_distance_idx]
    
    return fpr, tpr, thresholds, roc_auc, optimal_threshold

def infer(benign_test_losses, mal_test_losses, THRESHOLD):
    
    inference_on_mal_test_data = [int(x >= THRESHOLD) for x in mal_test_losses]
    inference_on_benign_test_data = [int(x >= THRESHOLD) for x in benign_test_losses]

    predicted = inference_on_mal_test_data + inference_on_benign_test_data
    actual = [1] * len(mal_test_losses) + [0] * len(benign_test_losses)

    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)

    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()

    total_positives = len(mal_test_losses)
    total_negatives = len(benign_test_losses)
    tp_rate = (tp / total_positives) * 100  
    tn_rate = (tn / total_negatives) * 100  
    fp_rate = (fp / total_negatives) * 100  
    fn_rate = (fn / total_positives) * 100 

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'True Positives (TP): {tp} ({tp_rate:.2f}%)')
    print(f'True Negatives (TN): {tn} ({tn_rate:.2f}%)')
    print(f'False Positives (FP): {fp} ({fp_rate:.2f}%)')
    print(f'False Negatives (FN): {fn} ({fn_rate:.2f}%)')
    
    return accuracy, precision, recall, f1, tp_rate, tn_rate, fp_rate, fn_rate