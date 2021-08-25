# federated-learning-models
In this repo, we aim to predict the remaining useful life (RUL) of turbofan engines from the NASA turbofan engine
degradation simulation data set FD001, using federated learning models.

This repo is divided into four main segments:
1. `data` - contains the train and test data for all workers in the federated learning process
2. `extracted` - contains the train and test RUL prediction output of all workers from the FATE model
3. `dc_extracted` - contains the train and test RUL prediction output of all workers from the dc_federated model
4. `script` - contains the federated learning pipeline for the FATE model
5. `dc_turbofan_models` - contains the federated learning pipeline for the dc_federated model

## References
* FATE - https://github.com/FederatedAI/FATE
* * dc_federated - https://github.com/digicatapult/dc-federated
