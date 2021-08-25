# federated-learning-models
In this repo, we aim to predict the remaining useful life (RUL) of turbofan engines from the NASA turbofan engine
degradation simulation data set FD001, using federated learning models.

This repo is divided into four main segments:
1. `data` - contains the train and test data for all workers in the federated learning process
2. `extracted` - contains the train and test RUL prediction output of all workers from the FATE model
3. `dc_extracted` - contains the train and test RUL prediction output of all workers from the dc_federated model
4. `script` - contains the federated learning pipeline for the FATE model
5. `dc_turbofan_models` - contains the federated learning pipeline for the dc_federated model

## FATE
The model for federated Gradient Boosted Decision Tree (GBDT) can be found in `pipeline_homo_sbt_regression.py`, which runs the customised federated learning pipeline for both training and testing purposes.

`pipeline-upload.py` supports the modelling process by uploading the necessary data for each worker onto FATE.

The number of workers can be adjusted by updating the variable `num_parties` in both of these files.

Remember to update the config file `config_5.yaml` and `config_3.yaml` to reflect the correct number of guests and hosts, in line with the number of workers.

![alt text](https://github.com/cchanzl/federated-learning-models/blob/master/images/FATEpipeline.png)

## dc_federated


## References
* FATE - https://github.com/FederatedAI/FATE
* dc_federated - https://github.com/digicatapult/dc-federated
