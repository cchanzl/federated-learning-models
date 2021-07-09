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

from homo_nn._common_component import run_homo_nn_pipeline, dataset

remaining_sensors = ['sens2', 'sens3', 'sens4', 'sens7', 'sens8', 'sens11',
                     'sens12', 'sens13', 'sens15', 'sens17', 'sens20', 'sens21']

epochs = 10
specific_lags = [1, 2, 3, 4, 5, 10, 20]
nodes_per_layer = [32, 64, 128]
dropout = 0.05
activation = 'relu'
batch_size = 16
input_dim = len(remaining_sensors)

def main(config="../../config.yaml", namespace=""):
    homo_nn_0 = HomoNN(name="homo_nn_0", config_type = 'keras',
                       max_iter=epochs,
                       batch_size=batch_size)
    # homo_nn_0.add(Dense(units=nodes_per_layer[0], input_dim=12, activation=activation))
    # homo_nn_0.add(Dropout(dropout))
    # homo_nn_0.add(Dense(units=nodes_per_layer[1], activation=activation))
    # homo_nn_0.add(Dropout(dropout))
    # homo_nn_0.add(Dense(units=nodes_per_layer[2], activation=activation))
    # homo_nn_0.add(Dropout(dropout))
    homo_nn_0.add(Dense(units=152, activation=activation))
    homo_nn_0.compile(optimizer=optimizers.Adam(learning_rate=0.05),
                      #metrics=["MeanSquaredError"],
                      #loss="mean_absolute_error")
                      loss="mean_squared_error")
    run_homo_nn_pipeline(config, namespace, dataset.nasa, homo_nn_0, 1)
