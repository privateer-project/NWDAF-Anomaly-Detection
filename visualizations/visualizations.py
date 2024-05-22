import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_train_val_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()
    
def plot_scatter_plot_rec_loss(benign_data_mse_losses, mal_data_mse_losses):
    index_benign = list(range(len(benign_data_mse_losses)))
    index_mal = list(range(len(mal_data_mse_losses)))

    plt.figure(figsize=(10, 6))

    plt.scatter(index_benign, benign_data_mse_losses, color='blue', label='Benign')
    plt.scatter(index_mal, mal_data_mse_losses, color='red', label='')

    plt.title('Scatter Plot of Benign & Malicious MSE loss')
    plt.xlabel('batch num')
    plt.ylabel('MSE loss')
    plt.legend() 

    plt.show()
    
def plot_roc_curve(fpr, tpr, thresholds , roc_auc ):

    trace_roc = go.Scatter(x=fpr, y=tpr, mode='lines', 
        name='ROC curve (area = {:.2f})'.format(roc_auc),
        line=dict(color='orange', width=2),
        hoverinfo='text',
        hovertext=[f'Threshold: {thresh:.4f}' for thresh in thresholds])

    trace_diag = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                            name='Reference',
                            line=dict(color='blue', width=2, dash='dash'),
                            showlegend=False)

    layout = go.Layout(title='Receiver Operating Characteristic',
                    xaxis=dict(title='False Positive Rate'),
                    yaxis=dict(title='True Positive Rate'),
                    width=700, height=550)

    fig = go.Figure(data=[trace_diag, trace_roc], layout=layout)

    fig.show()
    
def plot_dist(benign_ts, mal_ts, title, nbins = 50, x_axis_range = [None, None]):
    df = pd.DataFrame({
        'Values': np.concatenate([benign_ts, mal_ts]),
        'Series': ['Benign'] * len(benign_ts) + ['Malicious'] * len(mal_ts)
    })

    fig = px.histogram(df, x='Values', color='Series', barmode='overlay', nbins = nbins,
                    title=title)

    fig.update_xaxes(range=x_axis_range)

    fig.show()
    
    
def plot_ts(ts1, ts2, ts1_name, ts2_name, range):

    trace1 = go.Scatter(
        x=np.arange(ts1.shape[0]),
        y=ts1,
        mode='lines',
        name=ts1_name,
        line=dict(color='red')
    )

    trace2 = go.Scatter(
        x=np.arange(ts2.shape[0]),
        y=ts2,
        mode='lines',
        name=ts2_name,
        line=dict(color='blue')
    )

    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)

    fig.update_layout(
        title='Time Series Plot of Two 1D Numpy Arrays',
        xaxis_title='Time',
        yaxis_title='Value',
        legend_title='Series'
    )
    
    fig.update_yaxes(range=range)

    fig.show()

def plot_original_vs_rec(ts_original, ts_rec, imeisv, metric):
    trace1 = go.Scatter(
        y = ts_original,
        mode = 'lines',
        name = 'Original Time Series'
    )

    trace2 = go.Scatter(
        y = ts_rec,
        mode = 'lines',
        name = 'Reconstructed Time Series'
    )

    layout = go.Layout(
        title = f'Time Series Comparison for imeisv {imeisv}',
        xaxis = {'title': 'Time Index'},
        yaxis = {'title': f'{metric}'}
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    fig.show()