# federated-learning-models
In this repo, we aim to predict the remaining useful life (RUL) of turbofan engines from the NASA turbofan engine
degradation simulation data set FD001, using federated learning models.

This repo is divided into four main segments:
1. `data` - contains the train and test data for all workers in the federated learning process
2. `script` - contains the federated learning pipeline for the FATE model
3. `extracted` - contains the train and test RUL prediction output of all workers from the FATE model
4. `dc_turbofan_models` - contains the federated learning pipeline for the dc_federated model
5. `dc_extracted` - contains the train and test RUL prediction output of all workers from the dc_federated model

## FATE
The model for federated Gradient Boosted Decision Tree (GBDT) can be found in `pipeline_homo_sbt_regression.py`, which runs the customised federated learning pipeline for both training and testing purposes.

`pipeline-upload.py` supports the modelling process by uploading the necessary data for each worker onto FATE.

The number of workers can be adjusted by updating the variable `num_parties` in both of these files.

Remember to update the config file `config_5.yaml` and `config_3.yaml` to reflect the correct number of guests and hosts, in line with the number of workers.

![alt text](https://github.com/cchanzl/federated-learning-models/blob/master/images/FATEpipeline.png)

## dc_federated
The model for federated Neural Network (NN) using the federated averaging (FedAvg) algorithm can be found in `turbofan_fed_model.py`.

Both `turbofan_fed_avg_server.py` and `turbofan_fed_avg_worker.py` make calls to the model.

An example call to the server is

```bash
python FATE-Ubuntu/dc_turbofan_models/turbofan_fed_avg_server.py --batch-size 24 --learn-rate 0.03 --iter-rounds 12 --layer-one 64 --layer-two 128 --layer-three 256 --drop-out 0 --acti-func sigmoid
```

An example call to each worker is 

```bash
python FATE-Ubuntu/dc_turbofan_models/turbofan_fed_avg_worker.py --server-host-ip 127.0.1.1 --server-port 8080 --batch-size 24 --learn-rate 0.03 --iter-rounds 12 --layer-one 64 --layer-two 128 --layer-three 256 --drop-out 0 --acti-func sigmoid --bal-imbal _balanced --party-code 5
```

The arguments supplied to each of the above command line call determines the hyperparameters that the FedAvg algorithm will be trained on.

This is the same set of commands that is being used in `run_optimisation.py` which executes a randomised grid search to tune the best performing federated NN model.

## References
* FATE - https://github.com/FederatedAI/FATE
* dc_federated - https://github.com/digicatapult/dc-federated
