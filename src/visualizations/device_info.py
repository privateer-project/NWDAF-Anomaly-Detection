import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from src.config import PathsConf

# Load the data
df = pd.read_csv(PathsConf.raw_dataset)
df = df.sample(10000)

# Convert _time to datetime format
df['_time'] = pd.to_datetime(df['_time'])

# Function to create the visualization
def create_device_monitoring_dashboard():
    # Create a Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Get unique devices
    unique_devices = df['imeisv'].unique()

    # Determine which devices are malicious
    malicious_devices = []
    benign_devices = []

    for device in unique_devices:
        device_rows = df[df['imeisv'] == device]
        if any(device_rows['malicious'] == 1):
            malicious_devices.append(device)
        else:
            benign_devices.append(device)

    print(f"Malicious devices: {len(malicious_devices)}")
    print(f"Benign devices: {len(benign_devices)}")

    # Get potential metrics for visualization
    # Focus on network performance metrics
    network_metrics = [
        'dl_bitrate', 'ul_bitrate', 'dl_tx', 'ul_tx', 'cqi', 'pusch_snr',
        'dl_mcs', 'ul_mcs', 'dl_retx', 'ul_retx', 'ul_path_loss', 'epre',
        'p_ue', 'ri', 'turbo_decoder_avg'
    ]

    available_metrics = [col for col in network_metrics if col in df.columns]

    # Define the app layout
    app.layout = html.Div([
        dbc.Card(
            dbc.CardBody([
                html.H1("Network Device Monitoring Dashboard", className="card-title"),
                html.Hr(),

                # Controls
                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Display Controls", className="card-title"),
                                dbc.Checklist(
                                    options=[
                                        {"label": "Show Malicious Devices", "value": "malicious"},
                                        {"label": "Show Benign Devices", "value": "benign"},
                                    ],
                                    value=["malicious", "benign"],  # Both options checked by default
                                    id="device-type-checklist",
                                    inline=True,
                                    switch=True,
                                ),
                                html.Br(),
                                html.Label("Select Metric:"),
                                dcc.Dropdown(
                                    id="metric-dropdown",
                                    options=[{"label": col.replace('_', ' '), "value": col}
                                             for col in available_metrics],
                                    value=available_metrics[0] if available_metrics else None,
                                    clearable=False,
                                ),
                            ]),
                            className="mb-3",
                        ),
                    ]),
                ]),

                # Plot
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id="timeseries-plot", style={"height": "600px"}),
                    ]),
                ]),

                # Legend
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.P([
                                "This visualization shows network metrics for each device over time. "
                                "Toggle the checkboxes to show/hide malicious or benign devices. "
                                "Select different metrics from the dropdown menu."
                            ]),
                            html.P([
                                html.Span("", style={"display": "inline-block", "width": "12px",
                                                     "height": "12px", "border-radius": "50%",
                                                     "background-color": "#ff6b6b", "margin-right": "5px"}),
                                " Malicious devices ",
                                html.Span("", style={"display": "inline-block", "width": "12px",
                                                     "height": "12px", "border-radius": "50%",
                                                     "background-color": "#51cf66",
                                                     "margin-right": "5px", "margin-left": "15px"}),
                                " Benign devices"
                            ]),
                            html.P([
                                "Background colors: ",
                                html.Span("Green", className="bg-success text-white px-2 rounded"),
                                " = Normal operation, ",
                                html.Span("Red", className="bg-danger text-white px-2 rounded ml-2"),
                                " = Attack period"
                            ]),
                        ], className="text-muted small mt-3"),
                    ]),
                ]),
            ]),
            className="shadow-sm",
        ),
    ], className="container-fluid p-4")

    # Define callback to update graph
    @app.callback(
        Output("timeseries-plot", "figure"),
        [
            Input("device-type-checklist", "value"),
            Input("metric-dropdown", "value"),
        ],
    )
    def update_timeseries(device_types, metric):
        # Create a new figure
        fig = go.Figure()

        try:
            # Set background to light green (normal operation)
            time_range = df['_time'].agg(['min', 'max'])

            # Handle possible empty dataframes
            if len(df) > 0 and metric in df.columns:
                y_min = df[metric].min() - (0.1 * abs(df[metric].min()) if df[metric].min() != 0 else 0.5)
                y_max = df[metric].max() + (0.1 * abs(df[metric].max()) if df[metric].max() != 0 else 0.5)
            else:
                y_min = -1
                y_max = 1

            fig.add_shape(
                type="rect",
                x0=time_range['min'],
                x1=time_range['max'],
                y0=y_min,
                y1=y_max,
                fillcolor="rgba(0, 255, 0, 0.1)",
                line=dict(width=0),
                layer="below"
            )

            # Check if there are any attack periods
            attack_periods = df[df['attack'] == 1]['_time'].unique()

            if len(attack_periods) > 0:
                # Group by time to identify attack periods
                time_groups = {}
                for _, row in df.iterrows():
                    time = row['_time']
                    if time not in time_groups:
                        time_groups[time] = {'attack': row['attack'] == 1}

                # Sort times
                times = sorted(time_groups.keys())

                # Find attack ranges
                start = None
                for i in range(len(times)):
                    time = times[i]
                    if time_groups[time]['attack'] and start is None:
                        start = time
                    elif not time_groups[time]['attack'] and start is not None:
                        # Add red background for attack period
                        fig.add_shape(
                            type="rect",
                            x0=start,
                            x1=times[i - 1],
                            y0=y_min,
                            y1=y_max,
                            fillcolor="rgba(255, 0, 0, 0.1)",
                            line=dict(width=0),
                            layer="below"
                        )
                        start = None

                # Handle ongoing attack at the end
                if start is not None:
                    fig.add_shape(
                        type="rect",
                        x0=start,
                        x1=times[-1],
                        y0=y_min,
                        y1=y_max,
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        line=dict(width=0),
                        layer="below"
                    )

            # Add traces for each device
            for device in unique_devices:
                device_df = df[df['imeisv'] == device]

                # Check if this device type should be displayed
                display_device = False

                if device in malicious_devices and "malicious" in device_types:
                    display_device = True
                    color = "#ff6b6b"  # Red for malicious
                    device_type = "malicious"
                elif device in benign_devices and "benign" in device_types:
                    display_device = True
                    color = "#51cf66"  # Green for benign
                    device_type = "benign"

                if display_device and len(device_df) > 0 and metric in device_df.columns:
                    # Sort by time
                    device_df = device_df.sort_values(by="_time")

                    # Add trace
                    fig.add_trace(go.Scatter(
                        x=device_df['_time'],
                        y=device_df[metric],
                        mode='lines+markers',
                        name=f"{str(device)[-4:]} ({device_type})",
                        line=dict(width=2, color=color),
                        marker=dict(size=4, color=color)
                    ))

            # Improve y-axis scaling for some metrics with large ranges
            if metric in ['dl_bitrate', 'ul_bitrate', 'bearer_0_dl_total_bytes', 'bearer_1_dl_total_bytes']:
                fig.update_layout(yaxis_type="log")

            # Update layout
            metric_name = metric.replace('_', ' ')
            fig.update_layout(
                title=f"{metric_name} Values Over Time",
                xaxis=dict(
                    title="Time",
                    # rangeslider=dict(visible=True),
                    type="date"
                ),
                yaxis=dict(
                    title=f"{metric_name}"
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="left",
                    x=0
                ),
                margin=dict(l=50, r=50, t=80, b=180),
                height=600
            )

            # Add a text message if no data is shown
            if not any(trace['mode'] == 'lines+markers' for trace in fig.data):
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="No data available for the selected filters",
                    showarrow=False,
                    font=dict(size=20)
                )

        except Exception as e:
            print(f"Error updating plot: {e}")
            # Create an empty figure with error message
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text=f"Error generating plot: {str(e)}",
                showarrow=False,
                font=dict(size=16, color="red")
            )

        return fig

    return app

# Run the app
if __name__ == '__main__':
    # Create the app
    app = create_device_monitoring_dashboard()
    print("Starting Dash server...")
    app.run_server(debug=True, port=8050)
    print("Dash server is running on http://127.0.0.1:8050/")