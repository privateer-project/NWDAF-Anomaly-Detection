import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.subplots import make_subplots

from config import MetaData


def plot_corr_matrix(corr_mat, x, y, title, xaxis_title, yaxis_title, output_path):
    corr_fig = go.Figure(data=go.Heatmap(z=corr_mat,
                                         x=x,
                                         y=y,
                                         zmin=-1,
                                         zmax=1,
                                         colorscale='RdBu',
                                         text=np.around(corr_mat, decimals=2),
                                         texttemplate='%{text}',
                                         textfont={"size": 10},
                                         hoverongaps=False)
                         )
    corr_fig.update_layout(title=title,
                           width=800,
                           height=800,
                           xaxis_title=xaxis_title,
                           yaxis_title=yaxis_title)
    corr_fig.update_xaxes(tickangle=45)
    corr_fig.update_yaxes(tickangle=-45)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    corr_fig.write_html(output_path)


metadata = MetaData()
csv_path = '/data/raw/amari_ue_data_merged_with_attack_number.csv'
raw_df = pd.read_csv(csv_path, parse_dates=['_time'])
raw_df = raw_df.astype(
    {feature: params.dtype for feature, params in metadata.features.items() if feature in raw_df.columns})

features = [feature for feature, params in metadata.features.items() if params.input and feature in raw_df.columns]

drop_features = [feature for feature, params in metadata.features.items() if params.drop and feature in raw_df.columns]
raw_df.drop(drop_features, axis='columns', inplace=True)
raw_df.dropna(thresh=2, axis='columns', inplace=True)


for feature in raw_df.columns:
    for device_id, device_params in metadata.devices.items():
        for process in metadata.features[feature].process:
            if process == 'delta':
                raw_df.loc[raw_df['imeisv'] == device_params.imeisv, feature] = raw_df.loc[
                    raw_df['imeisv'] == device_params.imeisv, feature].diff().fillna(0)

raw_df = raw_df.set_index('_time')
from_ts = pd.to_datetime('2024-08-20 04:00:00')
to_ts = pd.to_datetime('2024-08-20 12:20:00')
raw_df = raw_df[(raw_df.index < from_ts) | (raw_df.index > to_ts)]
from_ts = pd.to_datetime('2024-08-20 20:50:00')
to_ts = pd.to_datetime('2024-08-20 22:10:00')
raw_df = raw_df[(raw_df.index < from_ts) | (raw_df.index > to_ts)]

for feature in features:
    emw_sr = raw_df.groupby('imeisv')[features].transform(lambda x: (x - x.ewm(alpha=0.1).mean()) / x.ewm(alpha=0.1).std())
    raw_df[features] = raw_df[features] *  emw_sr / raw_df[features].max()

# Resample data
resampled_devices_dfs = []

output_dir = 'analysis_results'
attack_corr_update = {}
attack_correlations = {}
attack_correlations_df = pd.DataFrame()


for device_id, device_params in metadata.devices.items():
    if device_params.malicious:
        device_type = 'Malicious'
    else:
        device_type = 'Benign'

    device_df = raw_df[raw_df['imeisv'] == device_params.imeisv].copy()

    resampled_device_df = device_df.resample('1min').agg({
        **{feat: 'mean' for feat, params in metadata.features.items() if params.input == True},
        'attack': 'max',
        'attack_number': 'max',
        'imeisv': 'first'
    })
    resampled_devices_dfs.append(resampled_device_df)

    device_corr_path = os.path.join(output_dir, f'{device_type}-{device_params.imeisv}', 'corr_mat.html')
    if not os.path.exists(device_corr_path):
        resampled_device_corr = resampled_device_df.corr()
        plot_corr_matrix(corr_mat=resampled_device_corr,
                         x=resampled_device_corr.columns,
                         y=resampled_device_corr.columns,
                         title=f'Correlation Matrix - {device_type} Device: {device_params.imeisv}',
                         xaxis_title='Features',
                         yaxis_title='Features',
                         output_path=device_corr_path)
    device_benign_df = resampled_device_df.loc[resampled_device_df['attack_number'] == '0']
    for attack_id, attack_params in metadata.attacks.items():
        device_attack_df = resampled_device_df.loc[attack_params.start: attack_params.stop]
        device_attack_df = pd.concat([device_attack_df, device_benign_df])  # Add benign traffic
        device_attack_corr_path = os.path.join(output_dir, f'{device_type}-{device_params.imeisv}',
                                               f'attack-{attack_id}-corr_mat.html')
        device_attack_corr = device_attack_df.corr()
        attack_corr_update[attack_id] = device_attack_corr['attack']


        if not os.path.exists(device_attack_corr_path):
            plot_corr_matrix(corr_mat=device_attack_corr,
                             x=device_attack_corr.columns,
                             y=device_attack_corr.columns,
                             title=f'Correlation Matrix - {device_type} Device: {device_params.imeisv} - attack {attack_id}',
                             xaxis_title='Features',
                             yaxis_title='Features',
                             output_path=device_attack_corr_path)
    attack_correlations_df[device_params.imeisv] = attack_corr_update
resampled_df = pd.concat(resampled_devices_dfs) if resampled_devices_dfs else pd.DataFrame(resampled_devices_dfs)

if resampled_df.empty:
    print(f'Empty resampled_df: {resampled_df}')
    exit()
titles = []
for device_id, device_params in metadata.devices.items():
    title = 'Malicious' if device_params.malicious else 'Benign'
    titles.append('-'.join([title, device_params.imeisv]))

for feature in features:
    fig = make_subplots(rows=len(metadata.devices), cols=1,
                        subplot_titles=titles,
                        shared_xaxes=True,
                        shared_yaxes=True,
                        vertical_spacing=0.05)
    fig.update_annotations(yshift=20)

    device_feature_stl_path = os.path.join(output_dir, 'features', f'{feature}.html')
    os.makedirs(os.path.dirname(device_feature_stl_path), exist_ok=True)
    if os.path.exists(device_feature_stl_path):
        continue

    for device_id, device_params in metadata.devices.items():
        device_feature_sr = resampled_df.loc[resampled_df['imeisv'] == device_params.imeisv, feature]

        # Get feature data and perform STL decomposition
        if len(device_feature_sr) > 0:
            # Create visualization
            fig.add_trace(
                go.Scatter(x=device_feature_sr.index, y=device_feature_sr.values, name='Original', marker_color='darkmagenta'),
                row=int(device_id), col=1)

    for attack_id, attack_params in metadata.attacks.items():
        fig.add_vrect(x0=pd.to_datetime(attack_params.start),
                      x1=pd.to_datetime(attack_params.stop),
                      fillcolor="red",
                      opacity=0.2,
                      line_width=0,
                      row='all',
                      col='all')

    fig.update_layout(height=2000,
                      title_text=f"{feature}",
                      title_x=0.5,
                      showlegend=False)
    fig.write_html(device_feature_stl_path)
    # Save plot

attack_corr_fig = make_subplots(rows=len(metadata.attacks),
                                cols=1,
                                subplot_titles=[f'Attack {key}' for key in metadata.attacks],
                                vertical_spacing=0.1,
                                horizontal_spacing=0.01,
                                shared_yaxes='all')
attack_corr_fig.update_annotations(yshift=20)
attack_cor_path = os.path.join(output_dir, 'attack_corr', 'attack-corr-per-device.html')
os.makedirs(os.path.dirname(attack_cor_path), exist_ok=True)

for i, group in attack_correlations_df.groupby(attack_correlations_df.index):
    attack_df = DataFrame()
    devices = []
    for device_id, device_params in metadata.devices.items():
        attack_df[device_params.imeisv] = group[device_params.imeisv].values[0]
        devices.append('Benign-' + device_params.imeisv if not device_params.malicious else 'Malicious-' + device_params.imeisv)

    attack_corr_fig.add_trace(go.Heatmap(z=attack_df.T,
                                         x=attack_df.index,
                                         y=devices,
                                         zmin=-1,
                                         zmax=1,
                                         colorscale='RdBu',
                                         text=np.around(attack_df.T, decimals=2),
                                         texttemplate='%{text}',
                                         textfont={"size": 10},
                                         hoverongaps=False),
                              row=i,
                              col=1)
attack_corr_fig.update_layout(height=2000,
                              title_text=f"Features per attack correlation",
                              title_x=0.5,
                              showlegend=False)

attack_corr_fig.write_html(attack_cor_path)
