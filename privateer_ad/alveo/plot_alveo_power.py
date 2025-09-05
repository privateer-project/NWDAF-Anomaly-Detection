import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List


def extract_alveo_power_data(
    data: List[Dict[str, Any]], xrt_version: str = "2.13"
) -> pd.DataFrame:
    """
    Extracts specified power data from a list of dictionaries, converts time to timestamps,
    and returns a sorted DataFrame.

    Args:
        data (list): A list of dictionaries containing power data.

    Returns:
        pd.DataFrame: A sorted DataFrame containing 'timestamp', 'Power', '12 Volts Auxillary Power',
                      '12 Volts PCI Express Power', and 'Internal FPGA Vcc Power' for each entry.
    """

    extracted_data = []
    for entry in data:
        # Convert time into a timestamp
        timestamp = entry["timestamp"]
        power = entry["Power"]

        # Compute specific powers
        power_12_volts_auxillary = (
            entry["Power Rails"]["12 Volts Auxillary"]["Voltage"]
            * entry["Power Rails"]["12 Volts Auxillary"]["Current"]
        )
        power_12_volts_pci_express = (
            entry["Power Rails"]["12 Volts PCI Express"]["Voltage"]
            * entry["Power Rails"]["12 Volts PCI Express"]["Current"]
        )
        power_internal_fpga_vcc = (
            entry["Power Rails"]["Internal FPGA Vcc"]["Voltage"]
            * entry["Power Rails"]["Internal FPGA Vcc"]["Current"]
        )

        # Append the extracted and calculated values to the list
        extracted_data.append(
            {
                "time": timestamp,
                "Power": power,
                "12 Volts Auxillary Power": power_12_volts_auxillary,
                "12 Volts PCI Express Power": power_12_volts_pci_express,
                "Internal FPGA Vcc Power": power_internal_fpga_vcc,
            }
        )

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(extracted_data)

    # Sort the DataFrame by ascending timestamp
    df = df.sort_values(by="time").reset_index(drop=True)

    return df


def plot_alveo_power_data(df, save_path: str = None) -> None:
    """
    Plots ALVEO power data from the DataFrame without dots, using only lines, ensures the y-axis includes zero,
    and saves the plot as a PNG file.

    Args:
        df (pd.DataFrame): A DataFrame containing the power data with timestamps.
        save_path (str): The file path to save the plot as a PNG file. If None, the plot is not saved. Defaults to None.
    """
    plt.figure(figsize=(12, 6))

    # Plotting the power data with lines only
    plt.plot(df["time"], df["Power"], label="ALVEO Total Power")
    plt.plot(
        df["time"], df["12 Volts Auxillary Power"], label="12 Volts Auxillary Power"
    )
    plt.plot(
        df["time"], df["12 Volts PCI Express Power"], label="12 Volts PCI Express Power"
    )
    plt.plot(df["time"], df["Internal FPGA Vcc Power"], label="Internal FPGA Vcc Power")

    # Ensuring that the y-axis includes zero
    # plt.axhline(0, color='gray', linestyle='--')  # Draws a line at y=0
    plt.ylim(
        bottom=0
    )  # Optionally, you can set the bottom limit to zero to make sure the grid starts from zero

    # Formatting the plot
    plt.title("ALVEO Power Data Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)  # Enable grid lines for better visibility
    plt.tight_layout()

    # Save the plot as a PNG file
    if save_path is not None:
        plt.savefig(
            save_path, format="png", dpi=300
        )  # Saves the plot with high resolution

    # Show the plot
    plt.show()