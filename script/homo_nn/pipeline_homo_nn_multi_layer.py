#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import pathlib
import sys

from pipeline.component.homo_nn import HomoNN
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout

additional_path = pathlib.Path(__file__).resolve().parent.parent.resolve().__str__()
if additional_path not in sys.path:
    sys.path.append(additional_path)

from homo_nn._common_component import run_homo_nn_pipeline

remaining_sensors = ['sens2', 'sens3', 'sens4', 'sens7', 'sens8', 'sens11',
                     'sens12', 'sens13', 'sens15', 'sens17', 'sens20', 'sens21']

# https://stackoverflow.com/questions/41488279/neural-network-always-predicts-the-same-class

alpha = 0.001  # 0.005
nb_classes = 16
epochs = 35
nodes = [128, 256, 512]  # [128, 256, 512]
dropout = 0.15
activation = 'relu'  # relu
batch_size = 32
input_dim = len(remaining_sensors)*2  # multiply by two for mean and trend

def main(config="../../config.yaml", namespace=""):
    homo_nn_0 = HomoNN(name="homo_nn_0", config_type = 'keras',
                       max_iter=epochs,
                       batch_size=batch_size)
    homo_nn_0.add(Dense(units=nodes[0], input_dim=input_dim, activation=activation))
    homo_nn_0.add(Dropout(dropout))
    # homo_nn_0.add(Dense(units=nodes[1], activation=activation))
    # homo_nn_0.add(Dropout(dropout))
    # homo_nn_0.add(Dense(units=nodes[2], activation=activation))
    # homo_nn_0.add(Dropout(dropout))
    homo_nn_0.add(Dense(units=nb_classes, activation="softmax"))
    homo_nn_0.compile(optimizer=optimizers.Adam(learning_rate=alpha),
                      metrics=["accuracy"],
                      loss="sparse_categorical_crossentropy")  # sparse CC can be used on integers categories
    run_homo_nn_pipeline(config, namespace, homo_nn_0)
