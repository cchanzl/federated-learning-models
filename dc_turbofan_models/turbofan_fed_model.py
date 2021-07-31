"""
Contains MNIST dataset specfic extension of FedAvgModelTrainer in
MNISTModelTrainer + associtaed helper class.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import pandas as pd
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from dc_federated.algorithms.fed_avg.fed_avg_model_trainer import FedAvgModelTrainer


class TurbofanNet(nn.Module):
    """
    CNN for the Turbofan dataset.
    """
    def __init__(self):
        super(TurbofanNet, self).__init__()
        self.layer1 = nn.Linear(24, 16)
        # self.dropout1 = nn.Dropout2d(0.25)
        self.layer2 = nn.Linear(16, 32)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.layer3 = nn.Linear(32, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        # x = self.dropout1(x)
        x = self.layer3(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        output = self.output(x)
        # output = F.log_softmax(x, dim=1)
        return output


class TurbofanNetArgs(object):
    """
    Class to abstract the arguments for a TurbofanModelTrainer .
    """
    def __init__(self):
        self.batch_size = 10
        self.test_batch_size = 1000
        self.epochs = 14
        self.lr = 0.5
        self.gamma = 0.7
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = False

    def print(self):
        print(f"batch_size: {self.batch_size}")
        print(f"test_batch_size: {self.test_batch_size}")
        print(f"epochs: {self.epochs}")
        print(f"lr: {self.lr}")
        print(f"gamma: {self.gamma}")
        print(f"no_cuda: {self.no_cuda}")
        print(f"seed: {self.seed}")
        print(f"log_interval: {self.log_interval}")
        print(f"save_model: {self.save_model}")


class TurbofanSubSet(torch.utils.data.Dataset):
    """
    Represents a Turbofan dataset subset. In particular, torchvision provides a
    Turbofan Dataset. This class wraps around that to deliver only a subset of
    digits in the train and test sets.

    Parameters
    ----------

    mnist_ds: torchvision.MNIST
        The core mnist dataset object.

    digits: int list
        The digits to restrict the data subset to

    args: MNISTNetArgs (default None)
        The arguments for the training.

    input_transform: torch.transforms (default None)
        The transformation to apply to the inputs

    target_transform: torch.transforms (default None)
        The transformation to apply to the target
    """
    def __init__(self, mnist_ds, digits, args=None, input_transform=None, target_transform=None):

        self.digits = digits
        self.args = TurbofanNetArgs() if not args else args
        mask = np.isin(mnist_ds.targets, digits)
        self.data = mnist_ds.data[mask].clone()
        self.targets = mnist_ds.targets[mask].clone()
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Implementation of a function required to extend Dataset. Returns\
        the dataset item at the given index.

        Parameters
        ----------
        index: int
            The index of the input + target to return.

        Returns
        -------

        pair of torch.tensors
            The input and the target.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.input_transform is not None: img = self.input_transform(img)
        if self.target_transform is not None: target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Implementation of a function required to extend Dataset. Returns\
        the length of the dataset.

        Returns
        -------

        int:
            The length of the dataset.
        """
        return len(self.data)

    def get_data_loader(self, use_cuda=False, kwargs=None):
        """
        Returns a DataLoader object for this dataset


        Parameters
        ----------

        use_cuda: bool (default False)
            Whether to use cuda or not.

        kwargs: dict
            Paramters to pass on to the DataLoader


        Returns
        -------
        
        DataLoader:
            A dataloader for this dataset.
        """

        if kwargs is None:
            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        return torch.utils.data.DataLoader(
            self, batch_size=self.args.batch_size, shuffle=True, **kwargs)


class TurbofanModelTrainer(FedAvgModelTrainer):
    """
    Trainer for the Turbofan data for the FedAvg algorithm. Extends
    the FedAvgModelTrainer, and implements all the relevant domain
    specific logic.

    Parameters
    ----------

    args: TurbofanNetArgs (default None)
        The arguments for the TurbofanNet

    model: TurbofanNet (default None)
        The model for training.

    train_loader: DataLoader
        The DataLoader for the train dataset.

    test_loader: DataLoader
        The DataLoader for the test dataset.

    rounds_per_iter: int
        The number of rounds to use per train() call.

    round_type: str (default 'batches')
        How to measure the number of training iterations. Allowed
        values are 'batches' and 'epochs'
    """
    def __init__(
            self,
            args=None,
            model=None,
            train_loader=None,
            test_loader=None,
            rounds_per_iter=10,
            round_type='batches'):
        self.args = TurbofanNetArgs() if not args else args

        self.use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # default model
        self.model = TurbofanNet().to(self.device) if not model else model

        self.train_loader = \
            TurbofanSubSet.default_dataset(True).get_loader() if not train_loader else train_loader
        self.test_loader = \
            TurbofanSubSet.default_dataset(False).get_loader() if not test_loader else test_loader

        self.rounds_per_iter = rounds_per_iter
        self.round_type = round_type

        # for housekeepiing.
        self._train_batch_count = 0
        self._train_epoch_count = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.args.gamma)

    def stop_train(self, batch_idx, current_iter_epoch_start):
        """
        Whether to stop the train the current training loop based on
        self.round_per_iter and self.round_type

        Parameters
        ----------

        batch_idx: int
            The index of current batch in the current epoch.

        current_iter_epoch_start: int
            The epoch at the start of the current training round.

        Returns
        -------

        bool:
            True if the training should be stopped.
        """
        return batch_idx > self.rounds_per_iter if self.round_type == 'batches' else\
            self._train_epoch_count - current_iter_epoch_start == self.rounds_per_iter

    def train(self):
        """
        Run the training on self.model using the data in the train_loader,
        using the given optimizer for the given number of epochs.
        print the results.
        """
        yhat_epoch = []
        ytru_epoch = []
        current_iter_epoch_start = self._train_epoch_count
        self.model.train()
        stop_training = False
        while not stop_training:
            yhat_temp = []
            ytru_temp = []
            # the data is trained through x epochs, with each epoch going through y batches of grad descent
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                # print(target)
                # print(output.flatten().tolist())
                yhat_temp = [*yhat_temp, *output.flatten().tolist()]
                ytru_temp = [*ytru_temp, *target.flatten().tolist()]

                loss = F.mse_loss(output, target)  # loss function
                loss.backward()
                self.optimizer.step()
                if self._train_batch_count % self.args.log_interval == 0:
                    print(f"Train Epoch: {self._train_epoch_count}"
                          f" [{self._train_batch_count * len(data)}/{len(self.train_loader.dataset)}"
                          f"({100. * self._train_batch_count / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                self._train_batch_count += 1

                # housekeeping after a single epoch
                if self._train_batch_count >= len(self.train_loader):
                    self._train_epoch_count += 1
                    self._train_batch_count = 0
                    self.scheduler.step()

                if self.stop_train(batch_idx, current_iter_epoch_start):
                    yhat_epoch.append(yhat_temp)
                    yhat_epoch.append(ytru_temp)
                    df_hat = pd.DataFrame(yhat_epoch).T
                    df_hat.to_csv("FATE-Ubuntu/dc_extracted/yhat.csv")
                    stop_training = True
                    break

            # save results of each epoch
            yhat_epoch.append(yhat_temp)


    def test(self):
        """
        Run the test on self.model using the data in the test_loader and
        print the results.
        """
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.mse_loss(output, target)

        test_loss /= len(self.test_loader.dataset)

        print(f"\nTest set: Average loss: {test_loss:.4f}")

    def get_model(self):
        """
        Returns the model for this trainer.

        Returns
        -------

        MNISTNet:
            The model used in this trainer.
        """
        return self.model

    def load_model(self, model_file):
        """
        Loads a model from the given model file.

        Parameters
        -----------

        model_file: io.BytesIO or similar object
            This object should contain a serilaized model.
        """
        self.model = torch.load(model_file)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.args.gamma)

    def load_model_from_state_dict(self, state_dict):
        """
        Loads a model from the given model file.

        Parameters
        -----------

        state_dict: dict
            Dictionary of parameter tensors.
        """
        self.model.load_state_dict(state_dict)

    def get_per_session_train_size(self):
        """
        Returns the size of the training set  used to train the local model
        in each iteration for use within FedAvg algorithm. If the
        round_type command line argument is 'batches', returns the number of
        batches per iteration, otherwise returns the actual number of samples
        used.
        
        Returns
        -------

        int:
            The number of samples
        """
        if self.round_type == 'batches':
            return self.rounds_per_iter * self.args.batch_size
        else:
            return self.rounds_per_iter * len(self.train_loader) * self.args.batch_size