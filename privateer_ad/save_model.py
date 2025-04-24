from logging import INFO
import pickle
from pathlib import Path
from flwr.common.logger import log
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg

import torch
from collections import OrderedDict
# from privateer_ad.models import AttentionAutoencoder
# from privateer_ad.config import AttentionAutoencoderConfig

# model_keys = AttentionAutoencoder(AttentionAutoencoderConfig).state_dict().keys()

class FedAvgWithModelSaving(FedAvg):
    """This is a custom strategy that behaves exactly like
    FedAvg with the difference of storing of the state of
    the global model to disk after each round.
    """
    def __init__(self, save_path: str, model_keys, *args, **kwargs):
        self.save_path = Path(save_path)
        # Create directory if needed
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.model_keys = model_keys
        super().__init__(*args, **kwargs)

    def _save_global_model(self, server_round: int, parameters):
        """A new method to save the parameters to disk."""
        ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(self.model_keys, ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        filename = str(self.save_path/f"model_round_{server_round}.pt")
        torch.save(state_dict, filename)
        # data = {'globa_parameters': ndarrays}
        
        # with open(filename, 'wb') as h:
        #     pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)
        log(INFO, f"Checkpoint saved to: {filename}")    

    def evaluate(self, server_round: int, parameters):
        """Evaluate model parameters using an evaluation function."""
        # save the parameters to disk using a custom method
        self._save_global_model(server_round, parameters)

        # call the parent method so evaluation is performed as
        # FedAvg normally does.
        return super().evaluate(server_round, parameters)