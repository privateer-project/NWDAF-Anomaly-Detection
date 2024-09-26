import flwr as fl


# fl.common.logger.configure(
#     identifier="myFlowerExperiment", filename="flwr_server_log.txt"
# )


def main():
    # Create a strategy using FedAvg with default parameters
    strategy = fl.server.strategy.FedAvg()

    # Start Flower server with the strategy
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Server address
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
