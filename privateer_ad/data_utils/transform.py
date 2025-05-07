import os
from pathlib import Path

import joblib

import pandas as pd
from pandas import DataFrame
from datasets import Dataset
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flwr_datasets.partitioner import PathologicalPartitioner
from torch.utils.data import DataLoader

from privateer_ad.config import MetaData, PathsConf, setup_logger
from privateer_ad.data_utils.utils import check_existing_datasets, get_dataset_path

logger = setup_logger('transform')

class DataProcessor:
    """Main class orchestrating data transform and loading."""
    def __init__(self):
        self.metadata = MetaData()
        self.paths = PathsConf()
        self.scaler_path = os.path.join(self.paths.scalers, 'scaler.pkl')
        self.pca_path = os.path.join(self.paths.scalers, 'pca.pkl')
        self.pca = None
        self.scaler = None
        self.input_features = self.metadata.get_input_features()
        self.drop_features = self.metadata.get_drop_features()
        self.devices = self.metadata.devices
        self.partitioner = None

    def split_data(self, df, save_dir, train_size=0.8):
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for device, device_info in self.devices.items():
            device_df = df.loc[df['imeisv'] == int(device_info.imeisv)]
            logger.info(f'Before Device: {device}, attack length: {len(device_df[device_df["attack"] == 1])}'
                        f' benign length: {len(device_df[device_df["attack"] == 0])}')
            device_df.loc[device_df['attack_number'].isin(device_info.in_attacks), 'attack'] = 1
            device_df.loc[~device_df['attack_number'].isin(device_info.in_attacks), 'attack'] = 0
            logger.info(f'After Device: {device}, attack length: {len(device_df[device_df["attack"] == 1])} '
                        f'benign length: {len(device_df[device_df["attack"] == 0])}')
            df_train, df_tmp = train_test_split(device_df,
                                                train_size=train_size,
                                                stratify=device_df['attack_number'],
                                                random_state=42)
            df_val, df_test = train_test_split( df_tmp,
                                                test_size=0.5,
                                                stratify=df_tmp['attack_number'],
                                                random_state=42)
            train_dfs.append(df_train)
            val_dfs.append(df_val)
            test_dfs.append(df_test)
        df_train = pd.concat(train_dfs)
        df_val = pd.concat(val_dfs)
        df_test = pd.concat(test_dfs)
        for i, df in enumerate([df_train, df_val, df_test]):
            logger.info(f'Dataset {i} attack length: {len(df[df["attack"] == 1])} '
                        f'benign length: {len(df[df["attack"] == 0])}')

        datasets = {'train': df_train, 'val': df_val, 'test': df_test}
        [logger.info(f'{key} shape: {df.shape}') for key, df in datasets.items()]
        os.makedirs(save_dir, exist_ok=True)
        logger.info('Save datasets...')
        for key, df in datasets.items():
            df.to_csv(get_dataset_path(key), index=False)
        logger.info(f'Datasets saved at {save_dir}')

    def clean_data(self, df):
        df = df.drop(columns=self.drop_features)
        df = df.drop_duplicates()
        df = df.dropna(axis='rows')
        df.reset_index(drop=True, inplace=True)
        return df

    def setup_scaler(self, df):
        """
        Scaling and normalizing numeric values, imputing missing values, clipping outliers,
         and adjusting values that have skewed distributions.
        """
        benign_df = df[df['attack_number'] == 0].copy()
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)

        self.scaler = StandardScaler()
        self.scaler.fit(benign_df[self.input_features])
        joblib.dump(self.scaler, self.scaler_path)
        return self.scaler

    def load_scaler(self):
        if self.scaler is None:
            self.scaler =  joblib.load(self.scaler_path)
        return self.scaler

    def apply_scale(self, df):
        """Scale numeric features using standardization (mean=0, std=1)."""
        self.load_scaler()
        transformed = self.scaler.transform(df[self.input_features])
        df.loc[:, self.input_features] = transformed
        return df

    def setup_pca(self, df, n_components=10):
        """Reduce feature dimensions using PCA."""
        self.pca = PCA(n_components=n_components)
        benign_df = df[df['attack_number'] == 0].copy()
        self.pca.fit(benign_df[self.input_features])
        os.makedirs(os.path.dirname(self.pca_path), exist_ok=True)
        joblib.dump(self.pca, self.pca_path)
        return self.pca

    def load_pca(self):
        if self.pca is None:
            self.pca =  joblib.load(self.pca_path)
        return self.pca

    def apply_pca(self, df):
        self.load_pca()
        pca_features = self.pca.transform(df[self.input_features])
        pca_df = pd.DataFrame(pca_features, columns=self.pca.get_feature_names_out())
        df = df.drop(columns=self.input_features)
        df = df.assign(**pca_df.to_dict(orient='list'))
        return df

    def initialize_data_pipeline(self,
                                 dataset_path,
                                 save_dir,
                                 n_components: int = None,
                                 train_size: float = 0.8) -> None:
        """Prepare complete datasets from raw data."""
        check_existing_datasets()
        raw_df = pd.read_csv(dataset_path)
        logger.info(f'Raw shape: {raw_df.shape}')
        self.split_data(raw_df, save_dir=save_dir, train_size=train_size)
        self.preprocess_data('train',
                             use_pca=False,
                             n_components=n_components,
                             setup=True)

    def preprocess_data(self,
                        path: str | Path,
                        use_pca: bool=False,
                        n_components=None,
                        partition_id=0,
                        num_partitions=1,
                        setup=False) -> DataFrame:
        if setup:
            if path in ('val', 'test'):
                logger.error(f'Setup on {path} is not allowed.')
                raise ValueError(f'Setup on {path} is not allowed.')
            logger.warning(f'{"#" * 30} Scalers will be fitted on {path}. {"#" * 30}')
        if path in ('train', 'val', 'test'):
            path = get_dataset_path(path)
        try:
            df = pd.read_csv(path, low_memory=False)
        except FileNotFoundError:
            raise FileNotFoundError(f'File {path} not found.')
        df = self.get_partition(df, partition_id=partition_id, num_partitions=num_partitions)
        df = self.clean_data(df)
        if setup:
            self.setup_scaler(df)
        df = self.apply_scale(df)
        if setup:
            self.setup_pca(df, n_components=n_components)
        if use_pca:
            df = self.apply_pca(df)
        df = df.sort_values('_time').reset_index(drop=True)
        return df

    def get_partition(self, df: DataFrame, partition_id=0, num_partitions=1) -> DataFrame:
        """Partition data based on provided configuration."""
        logger.info(f'partition_id: {partition_id} - num_partitions: {num_partitions}')
        #  if name != 'support'
        if num_partitions == 1:
            num_classes_per_partition = 3
        elif num_partitions == 3:
            num_classes_per_partition = 1
        else:
             num_classes_per_partition = 3 // num_partitions

        if self.partitioner is None:
            # _time,imeisv,5g_tmsi,amf_ue_id,bearer_0_apn,bearer_0_dl_total_bytes,bearer_0_ip,bearer_0_ipv6,bearer_0_pdu_session_id,bearer_0_qos_flow_id,bearer_0_sst,bearer_0_ul_total_bytes,bearer_1_apn,bearer_1_dl_total_bytes,bearer_1_ip,bearer_1_pdu_session_id,bearer_1_qos_flow_id,bearer_1_sst,bearer_1_ul_total_bytes,dl_bitrate,ran_id,ran_plmn,ran_ue_id,registered,rnti,t3512,tac,tac_plmn,ue_aggregate_max_bitrate_dl,ue_aggregate_max_bitrate_ul,ul_bitrate,bearer_1_ipv6,cell,ul_retx,ul_err,ul_mcs,ul_n_layer,ul_path_loss,ul_phr,ul_rank,dl_err,dl_mcs,dl_retx,dl_tx,cqi,epre,initial_ta,p_ue,pusch_snr,ri,turbo_decoder_avg,turbo_decoder_max,turbo_decoder_min,ul_tx,cell_id,attack,malicious,attack_number
            self.partitioner = PathologicalPartitioner(
                num_partitions=num_partitions,
                num_classes_per_partition=num_classes_per_partition,
                partition_by='cell',
                class_assignment_mode='first-deterministic')
            self.partitioner.dataset = Dataset.from_pandas(df)
        partitioned_df = self.partitioner.load_partition(partition_id).to_pandas(batched=False)
        return partitioned_df[df.columns]

    def get_dataloader(self,
                       path,
                       use_pca,
                       batch_size,
                       seq_len,
                       partition_id=0,
                       num_partitions=1,
                       only_benign=False) -> DataLoader:

        """Get train, validation and test dataloaders."""
        dataloader_params = {'train': path == 'train',
                             'batch_size': batch_size,
                             'num_workers': os.cpu_count(),
                             'pin_memory': True,
                             'prefetch_factor': 10000,
                             'persistent_workers': True}

        logger.info('Loading datasets...')

        df = self.preprocess_data(path,
                                  use_pca,
                                  partition_id=partition_id,
                                  num_partitions=num_partitions,
                                  setup=False)
        if only_benign:
            if 'attack' in df.columns:
                df = df[df['attack'] == 0]
            else:
                logger.warning('Cannot get benign data. No column named `attack` in dataset.')
        df = df.sort_values(by=['_time'])
        if use_pca:
            input_columns = [column for column in df.columns if column.startswith('pca')]
        else:
            input_columns = self.input_features
        df['time_idx'] = df.groupby('imeisv')['_time'].cumcount()
        dataloader = TimeSeriesDataSet(data=df,
                                       time_idx='time_idx',
                                       target='attack',
                                       group_ids=['imeisv'],
                                       max_encoder_length=seq_len,
                                       max_prediction_length=1,
                                       time_varying_unknown_reals=input_columns,
                                       scalers=None,
                                       target_normalizer=None,
                                       allow_missing_timesteps=False
                                       ).to_dataloader(**dataloader_params)
        logger.info('Finished loading datasets.')
        return dataloader

if __name__ == '__main__':
    paths = PathsConf()
    dp = DataProcessor()
    dp.initialize_data_pipeline(dataset_path=paths.raw_dataset,
                                save_dir=paths.processed)
