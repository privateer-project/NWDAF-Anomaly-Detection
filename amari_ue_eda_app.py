import os

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

data_folder = "./Data/Data v5"

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    
    df = pd.read_csv(os.path.join(data_folder, "amari_ue_data_final_v5.csv"))
    df = df.sort_values(['imeisv','_time'], ascending = True)

    summary_df = pd.read_csv(os.path.join(data_folder, "summary_df.csv"))
    summary_df['IMEISV'] = summary_df['IMEISV'].apply(str)
    return summary_df, df


summary_df, df = load_data()


imeisvs = df['imeisv'].unique().tolist()
metrics = df.columns
#df['timestamp'] = pd.to_datetime(df['timestamp'])

st.title('Amarisoft UE Explanatory Data Analysis')

st.dataframe(summary_df)

select_all_option = "Select All"
with st.sidebar:
    selected_imeisv = st.multiselect('Select IMEISVs', options=[select_all_option] + imeisvs)
    selected_metrics = st.multiselect('Select Metrics', options=[select_all_option] + list(metrics))
    window_selection = st.selectbox('Select rolling window', options = [None, 5, 10, 15, 20, 25, 30, 120])

    if select_all_option in selected_imeisv:
        selected_imeisv = imeisvs

    if select_all_option in selected_metrics:
        selected_metrics = metrics


filtered_df = df[df['imeisv'].isin(selected_imeisv)]

for metric in selected_metrics:
    imeisv_df_for_ma = {}

    for imeisv, imeisv_df in filtered_df.groupby('imeisv'):
        if window_selection:
            imeisv_df[metric] = imeisv_df[metric].rolling(window=window_selection).mean()
        else:
            pass
            
        imeisv_df_for_ma[str(imeisv)] = imeisv_df
            
    imeisv_ma_df = pd.concat(list(imeisv_df_for_ma.values()))
            
    fig = px.line(imeisv_ma_df, x='_time', y=metric, color='imeisv',
                labels={'time': 'Time', metric: 'Value'},
                title=f'{metric} over Time by IMEISV and Label')
    
    fig.add_vline(x='2024-03-23 21:26:00', line_dash="dash", line_color="black")
    fig.add_vline(x='2024-03-23 22:23:00', line_dash="dash", line_color="black")
    fig.add_vline(x='2024-03-23 22:56:00', line_dash="dash", line_color="black")
    fig.add_vline(x='2024-03-23 23:56:00', line_dash="dash", line_color="black")

    fig.update_layout(height=600, width=1200)
    
    st.plotly_chart(fig)