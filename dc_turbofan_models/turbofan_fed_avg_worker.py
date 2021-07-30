"""
Simple runner to start FedAvgWorker for the MNIST dataset.
"""

import pandas as pd
import argparse
from dc_federated.algorithms.fed_avg.fed_avg_worker import FedAvgWorker
from turbofan_fed_model import TurbofanModelTrainer, TurbofanSubSet

def get_args():
    """
    Parse the argument for running the MNIST worker.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Run this with the parameter provided by running the mnist_fed_avg_server\n")

    p.add_argument("--server-protocol",
                   help="The protocol used by the server (http or https)",
                   type=str,
                   default=None,
                   required=False)
    p.add_argument("--server-host-ip",
                   help="The ip of the host of server",
                   type=str,
                   required=True)
    p.add_argument("--server-port",
                   help="The ip of the host of server",
                   type=str,
                   required=True)

    p.add_argument("--party-code",
                   help="The party data that should be assigned to this worker",
                   type=str,
                   required=True)

    p.add_argument("--round-type",
                   help="What defines a training round. Allowed values (batches, epochs)",
                   type=str,
                   default='epochs',
                   required=False)

    p.add_argument("--rounds-per-iter",
                   help="The number of rounds per iteration of training of the worker.",
                   type=int,
                   default=10,
                   required=False)

    p.add_argument("--private-key-file",
                   help="The number of rounds per iteration of training of the worker.",
                   type=str,
                   default=None,
                   required=False)

    return p.parse_args()


def run():
    """
    This should be run to start a FedAvgWorker. Run this script with the --help option
    to see what the options are.

    --party-code A corresponds to Data set A. Only accepts A, B or C.
    """

    args = get_args()

    train_file_name = 'FATE-Ubuntu/data/party_' + args.party_code + '_train.csv'
    test_file_name = 'FATE-Ubuntu/data/party_' + args.party_code + '_test.csv'

    df_train = pd.read_csv(train_file_name)
    df_test = pd.read_csv(test_file_name)

    # https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
    import torch
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    import torch.utils.data as data_utils
    train_inputs = df_train[['x0', 'x1', 'x2']].astype(np.float32)
    train_target = df_train['y'].astype(np.float32)

    test_inputs = df_test[['x0', 'x1', 'x2']].astype(np.float32)
    test_target = df_test['y'].astype(np.float32)

    inputs = torch.tensor(train_inputs.values)
    targets = torch.tensor(train_target.values)
    train_dataset = TensorDataset(inputs, targets)
    train_data_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    inputs = torch.tensor(test_inputs.values)
    targets = torch.IntTensor(test_target.values)
    test_dataset = TensorDataset(inputs, targets)
    test_data_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    local_model_trainer = TurbofanModelTrainer(
        train_loader=train_data_loader,
        test_loader=test_data_loader,
        round_type=args.round_type,
        rounds_per_iter=args.rounds_per_iter
    )

    fed_avg_worker = FedAvgWorker(fed_model_trainer=local_model_trainer,
                                  private_key_file=args.private_key_file,
                                  server_protocol=args.server_protocol,
                                  server_host_ip=args.server_host_ip,
                                  server_port=args.server_port)
    fed_avg_worker.start()


if __name__ == '__main__':
    run()
