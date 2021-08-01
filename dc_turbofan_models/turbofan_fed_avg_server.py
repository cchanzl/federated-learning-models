"""
Simple runner to start FedAvgServer server for the MNIST dataset.
"""
import argparse
import pandas as pd
import sys
import torch.nn as nn
from turbofan_fed_model import TurbofanModelTrainer, TurbofanNetArgs, TurbofanNet
from dc_federated.algorithms.fed_avg.fed_avg_server import FedAvgServer

# to enable printing to log
import logging
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

def get_args():
    """
    Parse the argument for running Turbofan server.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Parameters for running the turbofan_avg_server\n")

    p.add_argument("--key-list-file",
                   help="The list of public keys for each worker to be authenticated.",
                   type=str,
                   required=False,
                   default=None)
    p.add_argument("--ssl-enabled", dest="ssl_enabled",
                   default=False, action="store_true")
    p.add_argument("--ssl-keyfile",
                   help="The path to the SSL key file.",
                   type=str,
                   required=False,
                   default=None)
    p.add_argument("--ssl-certfile",
                   help="The path to the SSL Certificate.",
                   type=str,
                   required=False,
                   default=None)
    p.add_argument("--server-host-ip",
                   help="The hostname or ip address of the server.",
                   type=str,
                   required=False,
                   default=None)
    p.add_argument("--server-port",
                   help="The port at which the server listens.",
                   type=int,
                   required=False,
                   default=8080)
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

    args, rest = p.parse_known_args()

    # We remove the known args because gunicorn also uses its own ArgumentParser that would conflict with this
    sys.argv = sys.argv[:1] + rest

    return args


def run():
    """
    This should be run to start the global server to test the backend + model integration
    in a distributed setting. Once the server has started, run the corresponding local-model
    runner(s) in local_model.py in other devices. Run `python federated_local_model.py -h' for instructions
    on how to do so.
    """
    args = get_args()

    df_train = pd.read_csv("FATE-Ubuntu/data/party_A_train.csv")
    df_test = pd.read_csv("FATE-Ubuntu/data/party_A_test.csv")

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
    train_data_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    inputs = torch.tensor(test_inputs.values)
    targets = torch.tensor(test_target.values)
    test_dataset = TensorDataset(inputs, targets)
    test_data_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # set hyperparameters
    model_args = TurbofanNetArgs()
    model_args.batch_size = args.batch_size
    model_args.epochs = args.epoch_no
    model_args.lr = args.learn_rate

    # define model
    model = TurbofanNet()
    model.layer1 = nn.Linear(24, args.layer_one)
    model.layer2 = nn.Linear(args.layer_one, args.layer_two)
    model.layer3 = nn.Linear(args.layer_two, args.layer_three)
    model.output = nn.Linear(args.layer_three, 1)


    # need to pass dataloader object into TurbofanModelTrainer()
    global_model_trainer = TurbofanModelTrainer(
        model=model,
        args=model_args,
        train_loader=train_data_loader,
        test_loader=test_data_loader,
    )

    fed_avg_server = FedAvgServer(global_model_trainer=global_model_trainer,
                                  key_list_file=args.key_list_file,
                                  update_lim=3,  # how many parties need to send info before global update starts
                                  server_host_ip=args.server_host_ip,
                                  server_port=args.server_port,
                                  ssl_enabled=args.ssl_enabled,
                                  ssl_keyfile=args.ssl_keyfile,
                                  ssl_certfile=args.ssl_certfile)
    fed_avg_server.start()


if __name__ == '__main__':
    run()
