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
    
    classic_df = pd.read_csv(os.path.join(data_folder, "enb_counters_data_classic_labeled.csv")).sort_values(['_time'], ascending = True)
    mini_df = pd.read_csv(os.path.join(data_folder, "enb_counters_data_mini_labeled.csv")).sort_values(['_time'], ascending = True)
    
    return classic_df, mini_df



classic_df, mini_df = load_data()

st.title('ENB Counters Explanatory Data Analysis')

with st.sidebar:
    selected_ds = st.selectbox('Select enb', options = ['classic', 'mini'])

    if selected_ds == 'mini':
        df = mini_df
        metrics = list(df.columns)
    else:
        df = classic_df
        metrics = list(df.columns)

    selected_metrics = st.multiselect('Select Metrics', options = metrics)
    window_selection = st.selectbox('Select rolling window', options = [None, 5, 10, 15, 20, 25, 30, 120])


for metric in selected_metrics:
    if window_selection:
        df[metric] = df[metric].rolling(window=window_selection).mean()
        
    fig = px.line(df, x='_time', y=metric,labels={'time': 'Time', metric: 'Value'}, title=f'{metric} over Time')
    
    fig.add_vline(x='2024-03-23 21:26:00', line_dash="dash", line_color="black")
    fig.add_vline(x='2024-03-23 22:23:00', line_dash="dash", line_color="black")
    fig.add_vline(x='2024-03-23 22:56:00', line_dash="dash", line_color="black")
    fig.add_vline(x='2024-03-23 23:56:00', line_dash="dash", line_color="black")

    fig.update_layout(height=600, width=1200)
    
    st.plotly_chart(fig)