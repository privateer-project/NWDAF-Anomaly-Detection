import numpy as np
import pandas as pd
import copy
import os
import time

from typing import List
from datetime import datetime

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import art
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox, MembershipInferenceBlackBoxRuleBased

from pytorch_forecasting import TimeSeriesDataSet
from dataclasses import dataclass, field
from opacus.layers import DPMultiheadAttention
import torch
from torch import nn
from privateer_ad.models.custom_layers.positional_encoding import PositionalEncoding
from privateer_ad.config.data_config import MetaData
from privateer_ad.config.hparams_config import HParams, AttentionAutoencoderConfig


from privateer_ad.utils import set_config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TAIMextended(nn.Module):
    """Attention-based autoencoder for anomaly detection."""

    def __init__(self, config: AttentionAutoencoderConfig, threshold=0.025):
        super(TAIMextended, self).__init__()
        self.config = config
        self.input_size = self.config.input_size
        self.hidden_dim = self.config.hidden_dim
        self.latent_dim = self.config.latent_dim
        self.dropout = self.config.dropout
        self.num_heads = self.config.num_heads
        self.num_layers = self.config.num_layers
        self.seq_len = self.config.seq_len

        self.optimal_threshold = threshold  # NOTE: need to deifine this maunally!
        self.criterion = nn.L1Loss(reduction="none")

        self.embed = nn.Linear(self.input_size, self.hidden_dim)
        self.pos_enc = PositionalEncoding(d_model=self.hidden_dim,
                                          max_seq_length=self.seq_len,
                                          dropout=self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.latent_dim,
            batch_first=True
        )
        encoder_layer.self_attn = DPMultiheadAttention(
            self.hidden_dim,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_dim)
        )

        self.compress = nn.Sequential(nn.Linear(self.hidden_dim, self.latent_dim),
                                    nn.ReLU())
        self.output = nn.Linear(self.latent_dim, self.input_size)

    def forward(self, x):
        x = x.reshape(-1, 12, 8)
        emb = self.embed(x)
        pos_enc = self.pos_enc(emb)
        enc = self.transformer_encoder(pos_enc)
        comp = self.compress(enc)
        decoded = self.output(comp)
        out = self.criterion(decoded, x).mean(dim=(1,2))
            
        return out


def taim_ext(sd, threshold=0.025, device=device):
    '''
    Load TAIM model that also give predictions. 
    Prediction of 1 indicates a malicious observation,
    while a predction of 0 indicates benign behavior.
    --------------------------------------------------
    sd:
        State dictionary of the model.
    device:
        CPU (cpu) or GPU (cuda) device. 
    '''
    taim_extended = TAIMextended(config=AttentionAutoencoderConfig(),threshold=threshold)
    taim_extended.load_state_dict(sd, strict=False)
    taim_extended.to(device)
    taim_extended.eval()
    return taim_extended

def sep_data_for_mi(
    data,
    labels,
    used_to_train=True
):
    '''
    Seperate data into two parts: 
        * One that has been used to train the AE
        * and one that the AE has not seen before
    Will also split this again into a traning and test
    set that can be used in Membership Inference attacks.
    '''
    all_inds_trained = np.arange(0,len(labels),1)
    rand_inds = np.random.choice(all_inds_trained, size=int(len(labels)/2), replace=False)
    # Select random inds to split data
    seperator = np.ones((len(labels)), dtype=bool)
    seperator[rand_inds] = False
    
    # Used to train MI model
    x_train_model = data[seperator].flatten(1)
    y_train_model = torch.zeros(x_train_model.shape[0])
    # Used to test MI model
    x_test_model = data[seperator == False].flatten(1)
    y_test_model = torch.zeros(x_test_model.shape[0])

    if used_to_train:
        y_train_model += 1
        y_test_model += 1

    return x_train_model, y_train_model, x_test_model, y_test_model


def mi_attack(
        model, 
        attack, 
        train_memb, 
        train_non_memb, 
        test_memb, 
        test_non_memb,
        save_path="results/test"):
    '''
    Performs a Membership inference attack and reports results
    '''
    art_model = art.estimators.regression.pytorch.PyTorchRegressor(
        model, 
        loss=torch.nn.MSELoss(), 
        input_shape=(12*8,)
        )
    bb_attack = MembershipInferenceBlackBox(art_model, input_type="loss", attack_model_type=attack)
    bb_attack.fit(train_memb, np.zeros(train_memb.shape[0]), 
                  train_non_memb, np.zeros(train_non_memb.shape[0]))

    # check performance
    # get inferred values
    inferred_train_bb = bb_attack.infer(test_memb, np.zeros(test_memb.shape[0]))
    inferred_test_bb = bb_attack.infer(test_non_memb, np.zeros(test_non_memb.shape[0]))

    member_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
    non_member_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
    acc = (member_acc * len(inferred_train_bb) + non_member_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
    print(f"Attack type: {attack}")
    print(f"Members Accuracy: {member_acc:.4f}")
    print(f"Non Members Accuracy {non_member_acc:.4f}")
    print(f"Attack Accuracy {acc:.4f}")


    np.savetxt(f"{save_path}_accs.txt", np.array([member_acc, non_member_acc, acc]), delimiter=",")


    y_train_pred = np.concatenate((inferred_train_bb, inferred_test_bb)) 
    y_train_true = np.concatenate((np.ones_like(inferred_train_bb), np.zeros_like(inferred_test_bb)))
    print(f"Black-Box model ({attack}):\n")
    print(classification_report(y_pred=y_train_pred, y_true=y_train_true))
    report = classification_report(y_pred=y_train_pred, y_true=y_train_true, output_dict=True)
    save_report_df = pd.DataFrame(report).transpose()
    save_report_df.to_csv(f"{save_path}.csv")

def attack_model_mi(
    state_dict,
    device,
    client_data_trained,
    client_data_not_trained,
    non_client_data_trained,
    non_client_data_not_trained,
    save_path="results/test"
):

    # Load model based on state_dict
    taim = taim_ext(state_dict, device=device)

    attack_types = ["rf","gb","knn"] 
    for attack in attack_types:
        # Initiate and train MembershipInference model (with RandomForest)
        mi_attack(
            taim, 
            attack,
            client_data_trained, 
            client_data_not_trained, 
            non_client_data_trained, 
            non_client_data_not_trained,
            f"{save_path}_{attack}")
        torch.cuda.empty_cache()

def load_data(df, used_to_train):

    ae_config = AttentionAutoencoderConfig()
    input_columns = MetaData().get_input_features()
    dataloader_params = {'train': False,
                         'batch_size': 4096,
                         'num_workers': os.cpu_count()-35,
                         'pin_memory': True,
                         'prefetch_factor': 10000,
                         'persistent_workers': True}
    
    dl = TimeSeriesDataSet(data=df,
                time_idx='time_idx',
                target='attack',
                group_ids=['cell_id'],
                # group_ids=['imeisv'],
                max_encoder_length=ae_config.seq_len,
                max_prediction_length=1,
                time_varying_unknown_reals=input_columns,
                scalers=None,
                target_normalizer=None,
                allow_missing_timesteps=False
                ).to_dataloader(**dataloader_params)

    data_list: List[float] = []
    label_list: List[float] = []
    for batch in dl:
        data_list.extend(batch[0]['encoder_cont'].cpu().tolist())
        label_list.extend(batch[1][0].cpu().tolist())

    del dl
    torch.cuda.empty_cache()
    data = np.array(data_list, dtype=np.float32).reshape(-1, ae_config.seq_len*len(input_columns))
    labels = np.array(label_list)[:,0]

    # If statement used to guarentee that there are no mal data
    if used_to_train:
        data = data[labels==0]

    return data


if __name__ == "__main__":

    # Data retrived by saving the loaded data from transform.py before it is put into DL
    data_type = "cell_tower"

    if data_type == "device":
        with open("data_device_split/our_df_with_id_0_and_path_id_train", "r") as f:
            df_train0 = pd.read_csv(f, index_col=0)
            f.close()  

        client_data_trained = load_data(df_train0, used_to_train=True)

        del df_train0
        print(client_data_trained.shape)
        


        with open("data_device_split/our_df_with_id_1_and_path_id_train", "r") as f:
            df_train1 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_trained1 = load_data(df_train1, used_to_train=True)
        del df_train1
        with open("data_device_split/our_df_with_id_2_and_path_id_train", "r") as f:
            df_train2 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_trained2 = load_data(df_train2, used_to_train=True)
        del df_train2
        with open("data_device_split/our_df_with_id_3_and_path_id_train", "r") as f:
            df_train3 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_trained3 = load_data(df_train3, used_to_train=True)
        del df_train3
        with open("data_device_split/our_df_with_id_4_and_path_id_train", "r") as f:
            df_train4 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_trained4 = load_data(df_train4, used_to_train=True)
        del df_train4
        with open("data_device_split/our_df_with_id_5_and_path_id_train", "r") as f:
            df_train5 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_trained5 = load_data(df_train5, used_to_train=True)
        del df_train5
        with open("data_device_split/our_df_with_id_6_and_path_id_train", "r") as f:
            df_train6 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_trained6 = load_data(df_train6, used_to_train=True)
        del df_train6
        with open("data_device_split/our_df_with_id_7_and_path_id_train", "r") as f:
            df_train7 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_trained7 = load_data(df_train7, used_to_train=True)
        del df_train7
        with open("data_device_split/our_df_with_id_8_and_path_id_train", "r") as f:
            df_train8 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_trained8 = load_data(df_train8, used_to_train=True)
        del df_train8
        # df_trained_data = pd.concat([df_train1,df_train2], ignore_index=True)
        
        
        non_client_data_trained = np.concatenate(
            (non_client_data_trained1, 
            non_client_data_trained2,
            non_client_data_trained3, 
            non_client_data_trained4,
            non_client_data_trained5, 
            non_client_data_trained6,
            non_client_data_trained7, 
            non_client_data_trained8),0)


        print(non_client_data_trained.shape)

        
        df_test0 = pd.read_csv("data_device_split/our_df_with_id_0_and_path_id_test", index_col=0)
        client_data_not_trained = load_data(df_test0, used_to_train=False)
        
        del df_test0


        with open("data_device_split/our_df_with_id_1_and_path_id_test", "r") as f:
                df_test1 = pd.read_csv(f, index_col=0)
                f.close()
        non_client_data_not_trained1 = load_data(df_test1, used_to_train=False)
        del df_test1
        with open("data_device_split/our_df_with_id_2_and_path_id_test", "r") as f:
            df_test2 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_not_trained2 = load_data(df_test2, used_to_train=False)
        del df_test2
        with open("data_device_split/our_df_with_id_3_and_path_id_test", "r") as f:
            df_test3 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_not_trained3 = load_data(df_test3, used_to_train=False)
        del df_test3
        with open("data_device_split/our_df_with_id_4_and_path_id_test", "r") as f:
            df_test4 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_not_trained4 = load_data(df_test4, used_to_train=False)
        del df_test4
        with open("data_device_split/our_df_with_id_5_and_path_id_test", "r") as f:
            df_test5 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_not_trained5 = load_data(df_test5, used_to_train=False)
        del df_test5
        with open("data_device_split/our_df_with_id_6_and_path_id_test", "r") as f:
            df_test6 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_not_trained6 = load_data(df_test6, used_to_train=False)
        del df_test6
        with open("data_device_split/our_df_with_id_7_and_path_id_test", "r") as f:
            df_test7 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_not_trained7 = load_data(df_test7, used_to_train=False)
        del df_test7
        with open("data_device_split/our_df_with_id_8_and_path_id_test", "r") as f:
            df_test8 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_not_trained8 = load_data(df_test8, used_to_train=False)
        del df_test8
            # df_non_trained_data = pd.concat([df_test1,df_test2], ignore_index=True)
            
        non_client_data_not_trained = np.concatenate((
            non_client_data_not_trained1, 
            non_client_data_not_trained2,
            non_client_data_not_trained3, 
            non_client_data_not_trained4,
            non_client_data_not_trained5, 
            non_client_data_not_trained6,
            non_client_data_not_trained7, 
            non_client_data_not_trained8),0)

        print(client_data_not_trained.shape)
        print(non_client_data_not_trained.shape)
    elif data_type == "cell_tower":
        with open("data_cell_tower_split/our_df_with_id_0_and_path_id_train", "r") as f:
            df_train0 = pd.read_csv(f, index_col=0)
            f.close()  

        client_data_trained = load_data(df_train0, used_to_train=True)

        del df_train0
        print(client_data_trained.shape)
        


        with open("data_cell_tower_split/our_df_with_id_1_and_path_id_train", "r") as f:
            df_train1 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_trained1 = load_data(df_train1, used_to_train=True)
        del df_train1
        with open("data_cell_tower_split/our_df_with_id_2_and_path_id_train", "r") as f:
            df_train2 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_trained2 = load_data(df_train2, used_to_train=True)
        del df_train2
        
        
        non_client_data_trained = np.concatenate(
            (non_client_data_trained1, 
            non_client_data_trained2),0)


        print(non_client_data_trained.shape)

        
        df_test0 = pd.read_csv("data_cell_tower_split/our_df_with_id_0_and_path_id_test", index_col=0)
        client_data_not_trained = load_data(df_test0, used_to_train=False)
        
        del df_test0


        with open("data_cell_tower_split/our_df_with_id_1_and_path_id_test", "r") as f:
                df_test1 = pd.read_csv(f, index_col=0)
                f.close()
        non_client_data_not_trained1 = load_data(df_test1, used_to_train=False)
        del df_test1
        with open("data_cell_tower_split/our_df_with_id_2_and_path_id_test", "r") as f:
            df_test2 = pd.read_csv(f, index_col=0)
            f.close()
        non_client_data_not_trained2 = load_data(df_test2, used_to_train=False)
        del df_test2
            
        non_client_data_not_trained = np.concatenate((
            non_client_data_not_trained1, 
            non_client_data_not_trained2),0)

        print(client_data_not_trained.shape)
        print(non_client_data_not_trained.shape)
    else:
        raise ValueError(f"data_type must be 'device' or 'cell_tower'. {data_type} was provided.")

    dp_levels = [0.0, 0.5, 1.0, 2.5, 5.0]  # noise_multiplier
    # dp_levels = [1.0, 2.5, 5.0]
    # dp_levels = [0.0, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
    max_grad_norm = [1,2.5,5,100]
    # dp_levels = [0.1]
    for dp in dp_levels:
        for mgn in max_grad_norm[::-1]:
            trial_id = datetime.now().strftime("%Y%m%d")
            # Run FL with flwr
            os.system(f'flwr run . --run-config "noise_multiplier={dp} max_grad_norm={mgn}"')
            
            # TODO: Should find path to different models automatically
            counter=0
            search = True
            while search and counter < 2:  # Also look into next day if unlucky
                try:
                    trial_id = list(trial_id)
                    trial_id[-1] = f"{int(trial_id[-1])+counter}"
                    trial_id = "".join(trial_id)
                    print(trial_id)
                    state_dict = torch.load(f"experiments/fl_train_noise_{dp}_max_grad_norm_{mgn}/{trial_id}/model_round_100.pt")
                    search = False
                except:
                    counter += 1 
                    continue
            # state_dict = torch.load(f"experiments/fl_train_noise_{dp}_max_grad_norm_{mgn}/{20250419}/model_round_10.pt")
            # state_dict = torch.load(f"experiments/fl_train/20250405-145838/model_round_1.pt")
            sd = {key.removeprefix('_module.'): value for key, value in state_dict.items()}

            
            client_data_trained_copy = copy.deepcopy(client_data_trained)        
            client_data_not_trained_copy = copy.deepcopy(client_data_not_trained)
            non_client_data_trained_copy = copy.deepcopy(non_client_data_trained)
            non_client_data_not_trained_copy = copy.deepcopy(non_client_data_not_trained)
            
            # attack model
            attack_model_mi(
            state_dict=sd,
            device=device,
            client_data_trained=client_data_trained_copy,
            client_data_not_trained=client_data_not_trained_copy,
            non_client_data_trained=non_client_data_trained_copy,
            non_client_data_not_trained=non_client_data_not_trained_copy,
            save_path=f"results/{data_type}_attack_50_epochs_10_aggs_target_epsilon_10/dp_level_noise_{dp}_max_grad_norm_{mgn}_target_epsilon_10")

            del client_data_trained_copy
            del client_data_not_trained_copy
            del non_client_data_trained_copy
            del non_client_data_not_trained_copy
            
        
