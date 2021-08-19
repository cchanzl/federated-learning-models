"""
Simple runner to start FedAvgWorker for the MNIST dataset.
"""

import pandas as pd
import argparse
import torch.nn as nn
from dc_federated.algorithms.fed_avg.fed_avg_worker import FedAvgWorker
from turbofan_fed_model import TurbofanModelTrainer, TurbofanNetArgs, TurbofanNet

def get_args():
    """
    Parse the argument for running the Turbofan worker.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Run this with the parameter provided\n")

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
                   type=int,
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

    p.add_argument("--batch-size",
                   help="Batch size",
                   type=int,
                   default=10,
                   required=False)

    p.add_argument("--learn-rate",
                   help="Learning rate",
                   type=float,
                   default=0.01,
                   required=False)

    p.add_argument("--epoch-no",
                   help="Number of epochs",
                   type=int,
                   default=10,
                   required=False)

    p.add_argument("--iter-rounds",
                   help="Number of iteration rounds per aggregation",
                   type=int,
                   default=7,
                   required=False)

    p.add_argument("--layer-one",
                   help="Number of iteration rounds per aggregation",
                   type=int,
                   default=16,
                   required=False)

    p.add_argument("--layer-two",
                   help="Number of iteration rounds per aggregation",
                   type=int,
                   default=32,
                   required=False)

    p.add_argument("--layer-three",
                   help="Number of iteration rounds per aggregation",
                   type=int,
                   default=64,
                   required=False)

    p.add_argument("--drop-out",
                   help="amount of dropout in layer 1 and 2",
                   type=float,
                   default=0.1,
                   required=False)

    p.add_argument("--acti-func",
                   help="Activation function used",
                   type=str,
                   default='tanh',
                   required=False)

    p.add_argument("--bal-imbal",
                   help="To use balanced or imbalanced dataset",
                   type=str,
                   default='',
                   required=False)

    return p.parse_args()


def run():
    """
    This should be run to start a FedAvgWorker. Run this script with the --help option
    to see what the options are.

    --party-code A corresponds to Data set A. Only accepts A, B or C.
    """

    args = get_args()

    train_file_name = 'FATE-Ubuntu/data/party_' + str(args.party_code) + '_train' + args.bal_imbal + '.csv'
    test_file_name = 'FATE-Ubuntu/data/party_' + str(args.party_code) + '_test' + args.bal_imbal + '.csv'

    df_train = pd.read_csv(train_file_name)
    df_test = pd.read_csv(test_file_name)

    # https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
    import torch
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader

    train_target = df_train.pop('y').astype(np.float32)
    df_train.pop('id')
    train_inputs = df_train.astype(np.float32)

    test_target = df_test.pop('y').astype(np.float32)
    df_test.pop('id')
    test_inputs = df_test.astype(np.float32)

    inputs = torch.tensor(train_inputs.values)
    targets = torch.tensor(train_target.values)
    train_dataset = TensorDataset(inputs, targets)
    train_data_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)

    inputs = torch.tensor(test_inputs.values)
    targets = torch.IntTensor(test_target.values)
    test_dataset = TensorDataset(inputs, targets)
    test_data_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # set hyperparameters
    model_args = TurbofanNetArgs()
    model_args.batch_size = args.batch_size
    model_args.epochs = args.epoch_no
    model_args.lr = args.learn_rate

    # define model
    model = TurbofanNet()
    model.activation = args.acti_func
    model.layer1 = nn.Linear(24, args.layer_one)
    model.dropout1 = nn.Dropout(args.drop_out)
    model.layer2 = nn.Linear(args.layer_one, args.layer_two)
    model.dropout2 = nn.Dropout(args.drop_out)
    model.layer3 = nn.Linear(args.layer_two, args.layer_three)
    model.output = nn.Linear(args.layer_three, 1)

    local_model_trainer = TurbofanModelTrainer(
        model=model,
        train_loader=train_data_loader,
        test_loader=test_data_loader,
        round_type=args.round_type,
        party='worker_' + str(args.party_code),
        rounds_per_iter=args.iter_rounds,
        args=model_args
    )

    fed_avg_worker = FedAvgWorker(fed_model_trainer=local_model_trainer,
                                  private_key_file=args.private_key_file,
                                  server_protocol=args.server_protocol,
                                  server_host_ip=args.server_host_ip,
                                  server_port=args.server_port)
    fed_avg_worker.start()


if __name__ == '__main__':
    run()
