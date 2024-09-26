# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
import ast

import pandas as pd
import numpy as np

from pathlib import Path

src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

# from dotenv import find_dotenv, load_dotenv
from functools import reduce
from datetime import datetime, timedelta, date

from configs.config import CFG


def parse_datetime(ctx, param, value):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise click.BadParameter(
            f"Invalid datetime format for '{value}'. Expected format: 'YYYY-MM-DD HH:MM:SS'."
        )


def parse_attacks(value):
    value = [*map(lambda x: ast.literal_eval(x), value)]
    attacks = []
    for attack in value:
        print(attack)
        try:
            start_str, end_str = attack
            start_dt = parse_datetime(None, None, start_str)
            end_dt = parse_datetime(None, None, end_str)
            attacks.append((start_dt, end_dt))
        except ValueError:
            raise click.BadParameter(
                f"Invalid attack format for '{attack}'. Expected format: 'start_date,end_date'."
            )
    return attacks


def process_amari_ue_raw_data(amari_ue_raw_data_path):
    amari_ue_data_df = pd.read_csv(amari_ue_raw_data_path, skiprows=3)

    amari_ue_data_df.drop(["Unnamed: 0", "result"], axis=1, inplace=True)
    amari_ue_data_df.dropna(how="all", inplace=True)
    amari_ue_data_df["imeisv"] = amari_ue_data_df["imeisv"].astype(str)
    amari_ue_data_df.drop(
        amari_ue_data_df[~(amari_ue_data_df["imeisv"].str.isnumeric())].index,
        inplace=True,
    )

    amari_ue_data_df["_time"] = pd.to_datetime(
        amari_ue_data_df["_time"], format="mixed"
    ).dt.tz_localize(None)

    imeisv_dfs = {}
    for imeisv, indices in amari_ue_data_df.groupby("imeisv").groups.items():
        imeisv_dfs[imeisv] = (
            amari_ue_data_df.loc[indices]
            .copy()
            .pivot(index=["_time", "imeisv"], columns="_field", values="_value")
            .reset_index()
        )

    amari_ue_data_ds = pd.concat(list(imeisv_dfs.values()), axis=0)
    return amari_ue_data_ds


def merge_amari_ue_data(amari_ue_classic_df, amari_ue_mini_df):
    merged_df = pd.concat(
        [amari_ue_classic_df, amari_ue_mini_df], axis=0, ignore_index=True
    )

    cell_1_cols = [*filter(lambda x: "cell_" in x, list(amari_ue_classic_df.columns))]
    cell_2_cols = [*filter(lambda x: "cell_" in x, list(amari_ue_mini_df.columns))]

    cell_1_metrics = [*map(lambda x: x.split("_"), cell_1_cols)]
    cell_2_metrics = [*map(lambda x: x.split("_"), cell_2_cols)]

    cell_1_metrics = [*map(lambda x: "_".join(x[x.index("1") + 1 :]), cell_1_metrics)]
    cell_2_metrics = [*map(lambda x: "_".join(x[x.index("2") + 1 :]), cell_2_metrics)]

    cell_metrics_dict = {
        cell_1_metric: cell_2_metric
        for cell_1_metric, cell_2_metric in zip(cell_1_cols, cell_2_cols)
    }

    i = 0
    for col1, col2 in cell_metrics_dict.items():
        merged_df[f"cell_x_{cell_1_metrics[i]}"] = np.where(
            pd.isnull(merged_df[col1]), merged_df[col2], merged_df[col1]
        )
        merged_df.drop(col1, axis=1, inplace=True)
        merged_df.drop(col2, axis=1, inplace=True)
        i += 1

    return merged_df


def decrement_cols(df):

    def decrement_cols_(df):
        df["ul_total_bytes_non_incr"] = df["ul_total_bytes"].diff().fillna(0)
        df["dl_total_bytes_non_incr"] = df["dl_total_bytes"].diff().fillna(0)

        return df.iloc[1:, :]

    df[["bearer_0_ul_total_bytes", "bearer_1_ul_total_bytes"]] = (
        df[["bearer_0_ul_total_bytes", "bearer_1_ul_total_bytes"]]
        .fillna(0)
        .astype(np.float)
    )

    print(df.dtypes)

    df["ul_total_bytes"] = df["bearer_0_ul_total_bytes"] + df["bearer_1_ul_total_bytes"]
    df["dl_total_bytes"] = df["bearer_0_dl_total_bytes"] + df["bearer_1_dl_total_bytes"]

    final_df = pd.concat(
        [decrement_cols_(imeisv_df) for _, imeisv_df in df.groupby("imeisv")]
    )

    return final_df


def label_data(df, attacks, malicious_imeisv):
    attacks_filters = []
    for attack_start, attack_end in attacks:
        attack_filter = df["_time"].between(attack_start, attack_end, inclusive="both")
        attacks_filters.append(attack_filter)

    combined_attacks_filter = reduce(lambda x, y: x | y, attacks_filters)
    malicious_imeisv_filter = df["imeisv"].isin(malicious_imeisv)

    df["label"] = np.where((combined_attacks_filter & malicious_imeisv_filter), 1, 0)
    return df


def split_data(
    df,
    benign_data_train_period_start,
    benign_data_train_period_end,
    imeisv_with_valid_benign_activity,
    benign_data_test_period_start,
    benign_devices_for_testing,
    first_attack_period_start,
    first_attack_period_end,
    second_attack_period_start,
    second_attack_period_end,
    imeisv_to_exclude_for_first_attack,
    imeisv_to_exclude_for_second_attack,
):
    benign_filter_1 = df["_time"].between(
        benign_data_train_period_start, benign_data_train_period_end
    )
    benign_filter_2 = ~df["imeisv"].isin(imeisv_with_valid_benign_activity)
    benign_filter_3 = df["label"] == 0
    benign_data_filter = benign_filter_1 & benign_filter_2 & benign_filter_3

    benign_data_train = df[benign_data_filter].copy()
    benign_data_train = benign_data_train.sort_values(["imeisv", "_time"])

    benign_filter_4 = df["_time"] >= benign_data_test_period_start
    benign_filter_5 = df["imeisv"].isin(benign_devices_for_testing)
    benign_data_filter_test = benign_filter_3 & benign_filter_4 & benign_filter_5

    benign_data_test = df[benign_data_filter_test].copy()
    benign_data_test = benign_data_test.sort_values(["imeisv", "_time"])

    mal_filter_1 = df["_time"].between(
        first_attack_period_start, first_attack_period_end
    ) & (~df["imeisv"].isin(imeisv_to_exclude_for_first_attack))

    mal_filter_2 = df["_time"].between(
        second_attack_period_start, second_attack_period_end
    ) & (~df["imeisv"].isin(imeisv_to_exclude_for_second_attack))

    mal_filter_3 = df["label"] == 1

    malicious_data = df[(mal_filter_1 | mal_filter_2) & mal_filter_3].copy()
    malicious_data = malicious_data.sort_values(["imeisv", "_time"])

    return benign_data_train, benign_data_test, malicious_data


@click.command()
# @click.argument(
#     "data_source",
#     type=click.Choice(["amari_ue", "enb_counters"], case_sensitive=False),
#     default=CFG["data"]["data_source"],
# )
@click.option(
    "--amari-ue-classic-input-filepath",
    type=click.Path(),
    default=CFG["data"]["amari_ue_classic_input_filepath"],
    help="Input file path for 'amari_ue classic'",
)
@click.option(
    "--amari-ue-mini-input-filepath",
    type=click.Path(),
    default=CFG["data"]["amari_ue_mini_input_filepath"],
    help="Input file path for 'amari_ue mini'",
)
@click.option(
    "--enb-counters-input-filepath",
    type=click.Path(),
    default=CFG["data"]["enb_counters_input_filepath"],
    help="Input file path for 'enb_counters'",
)
@click.option(
    "--amari-ue-output-filepath",
    type=click.Path(),
    default=CFG["data"]["amari_ue_output_filepath"],
    help="Output file path for 'amari_ue'",
)
@click.option(
    "--enb-counters-output-filepath",
    type=click.Path(),
    default=CFG["data"]["enb_counters_output_filepath"],
    help="Output file path for 'enb_counters'",
)
@click.option(
    "--attacks",
    type=(str, str),
    multiple=False,
    default=CFG["data"]["attacks"],
    help="List of tuples containing start and end datetime values in 'YYYY-MM-DD HH:MM:SS' format",
)
@click.option(
    "--malicious-imeisv",
    type=str,
    multiple=True,
    default=CFG["data"]["malicious_imeisv"],
    help="List of malicious IMEIs or phone numbers.",
)
def main(
    amari_ue_classic_input_filepath,
    amari_ue_mini_input_filepath,
    amari_ue_output_filepath,
    enb_counters_input_filepath,
    enb_counters_output_filepath,
    attacks,
    malicious_imeisv,
):
    """
    Main function that processes the input file(s) based on the data source and saves the result to the output file(s).
    Depending on the data source, prompts for additional inputs if not provided.

    DATA_SOURCE: The data source type ('amari_ue' or 'enb_counters').
    """

    parsed_attacks = parse_attacks(attacks)
    for start, end in parsed_attacks:
        click.echo(f"Attack period: {start} to {end}")

    if not amari_ue_classic_input_filepath:
        amari_ue_classic_input_filepath = click.prompt(
            "Please enter the input file path for 'classic'",
            type=click.Path(exists=True),
        )
    if not amari_ue_mini_input_filepath:
        amari_ue_mini_input_filepath = click.prompt(
            "Please enter the input file path for 'mini'",
            type=click.Path(exists=True),
        )
    if not amari_ue_output_filepath:
        amari_ue_output_filepath = click.prompt(
            "Please enter the output file path for 'amari_ue'",
            type=click.Path(exists=False),
        )

    click.echo(f"Processing amari_ue data")

    amari_ue_classic_df = process_amari_ue_raw_data(amari_ue_classic_input_filepath)
    amari_ue_mini_df = process_amari_ue_raw_data(amari_ue_mini_input_filepath)

    amari_ue_merged_df = merge_amari_ue_data(amari_ue_classic_df, amari_ue_mini_df)
    amari_eu_merged_labeled = label_data(
        amari_ue_merged_df, parsed_attacks, malicious_imeisv
    )

    dtype_mapping = {
        "dl_bitrate": float,
        "ul_bitrate": float,
        "cell_x_dl_retx": int,
        "cell_x_dl_tx": int,
        "cell_x_ul_retx": int,
        "cell_x_ul_tx": int,
        "bearer_0_ul_total_bytes": float,
        "bearer_1_ul_total_bytes": float,
        "bearer_0_dl_total_bytes": float,
        "bearer_1_dl_total_bytes": float,
    }

    amari_eu_merged_labeled = amari_eu_merged_labeled.astype(dtype_mapping)

    amari_eu_merged_labeled = decrement_cols(amari_eu_merged_labeled)
    amari_eu_merged_labeled.to_csv(amari_ue_output_filepath)

    benign_data_train, benign_data_test, malicious_data = split_data(
        amari_eu_merged_labeled,
        "2024-03-20 14:14:50.19",
        "2024-03-23 16:26:19.00",
        [
            "8642840401594200",
            "8642840401612300",
            "8642840401624200",
            "3557821101183501",
        ],
        "2024-03-24 01:20:00.19",
        ["8609960468879057", "8628490433231157", "8677660403123800"],
        "2024-03-23 21:26:00",
        "2024-03-23 22:23:00",
        "2024-03-23 22:56:00",
        "2024-03-23 23:56:00",
        ["8628490433231157", "8609960480666910", "3557821101183501"],
        ["8609960480666910", "8642840401612300"],
    )

    benign_data_train.to_csv(CFG["data"]["benign_train_path"])
    benign_data_test.to_csv(CFG["data"]["benign_test_path"])
    malicious_data.to_csv(CFG["data"]["malicious_path"])

    if not enb_counters_input_filepath:
        pass
        # enb_counters_input_filepath = click.prompt(
        #     "Please enter the input file path", type=click.Path(exists=True)
        # )
    if not enb_counters_output_filepath:
        pass
        # enb_counters_output_filepath = click.prompt(
        #     "Please enter the output file path", type=click.Path()
        # )

        # click.echo(f"Processing enb_counters data")

    print("Done")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
