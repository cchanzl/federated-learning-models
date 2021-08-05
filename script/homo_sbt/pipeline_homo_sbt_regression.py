#  python /main/script/homo_sbt/pipeline_homo_sbt_regression.py -config /main/script/homo_sbt/config.yaml
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

import argparse
import json
import numpy as np
import random
from pipeline.component import HomoDataSplit
from pipeline.backend.pipeline import PipeLine
from pipeline.component.data_transform import DataTransform
from pipeline.component.homo_secureboost import HomoSecureBoost
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.component.evaluation import Evaluation
from pipeline.interface.model import Model

from pipeline.utils.tools import load_job_config
from pipeline.runtime.entity import JobParameters


def main(config="../../config.yaml", namespace=""):
    # obtain config.yaml
    if isinstance(config, str):
        config = load_job_config(config)

    # read input from config.yaml
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host  # multiple other parties
    arbiter = parties.arbiter[0]

    backend = config.backend
    work_mode = config.work_mode

    party_A_train_data = {"name": "nasa_A", "namespace": f"experiment"}
    party_A_test_data = {"name": "nasa_A_test", "namespace": f"experiment"}

    party_B_train_data = {"name": "nasa_B", "namespace": f"experiment"}
    party_B_test_data = {"name": "nasa_B_test", "namespace": f"experiment"}

    party_C_train_data = {"name": "nasa_C", "namespace": f"experiment"}
    party_C_test_data = {"name": "nasa_C_test", "namespace": f"experiment"}

    job_parameters = JobParameters(backend=backend, work_mode=work_mode)
    filename = ""
    hyperparameter_config = []

    ITERATIONS = 1
    train = True
    if train:
        for iteration in range(ITERATIONS):

            # Set random grid search parameters
            # num_tree = int(random.sample(list(np.arange(30, 40, 5)), 1)[0])
            # lrn_rate = random.sample(list(np.arange(1, 20, 5)/100), 1)[0]
            # max_dept = int(random.sample(list(np.arange(5, 20, 1)), 1)[0])
            # val_freq = int(random.sample(list(np.arange(10, 30, 1)), 1)[0])

            num_tree = 30
            lrn_rate = 0.11
            max_dept = 5
            val_freq = 25

            print("Start training iteration " + str(iteration))
            print("num_tree: " + str(num_tree))
            print("lrn_rate: " + str(lrn_rate))
            print("max_dept: " + str(max_dept))
            print("val_freq: " + str(val_freq))

            # set job initiator
            pipeline = PipeLine().set_initiator(role='guest', party_id=guest)
            pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

            # 0 for train data, 1 for test data
            datatransform_0 = DataTransform(name="datatransform_0")
            reader_0= Reader(name="reader_0")

            reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=party_C_train_data)
            reader_0.get_party_instance(role='host', party_id=host[0]).component_param(table=party_B_train_data)
            reader_0.get_party_instance(role='host', party_id=host[1]).component_param(table=party_A_train_data)

            datatransform_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True,
                                                                                      output_format="dense",
                                                                                      label_type="float")
            datatransform_0.get_party_instance(role='host', party_id=host[0]).component_param(with_label=True,
                                                                                       output_format="dense",
                                                                                       label_type="float")
            datatransform_0.get_party_instance(role='host', party_id=host[1]).component_param(with_label=True,
                                                                                       output_format="dense",
                                                                                       label_type="float")

            homo_secureboost_0 = HomoSecureBoost(name="homo_secureboost_0",
                                                 learning_rate=lrn_rate,
                                                 num_trees=num_tree,  # 50 is best
                                                 task_type='regression',
                                                 # None,'cross_entropy','lse','lae','log_cosh','tweedie','fair','huber'
                                                 objective_param={"objective": "lse"},  # lse is best
                                                 tree_param={"max_depth": max_dept},
                                                 validation_freqs=val_freq)

            evaluation_0 = Evaluation(name='evaluation_0', eval_type='regression')
            pipeline.add_component(reader_0)
            pipeline.add_component(datatransform_0, data=Data(data=reader_0.output.data))
            # https://github.com/FederatedAI/FATE/tree/178f04d1a58181359d6550b4673d4b4dc72a778f/python/fate_client/pipeline/component
            # https://fate.readthedocs.io/en/latest/_build_temp/python/federatedml/README_zh.html?highlight=DataSplitParam#federatedml.param.DataSplitParam
            homo_data_split_1 = HomoDataSplit(name='homo_data_split_1', train_size=0.6, validate_size=0.4)
            pipeline.add_component(homo_data_split_1, data=Data(data=datatransform_0.output.data))
            pipeline.add_component(homo_secureboost_0, data=Data(train_data=homo_data_split_1.output.data.train_data,
                                                                 validate_data=homo_data_split_1.output.data.validate_data))
            pipeline.add_component(evaluation_0, data=Data(homo_secureboost_0.output.data))
            pipeline.compile()

            pipeline.fit(job_parameters)
            print(f"Train Evaluation summary:\n{json.dumps(pipeline.get_component('evaluation_0').get_summary(), indent=4)}")
            # json_string = json.dumps(pipeline.get_component('evaluation_0').get_summary())
            # json_dict = json.loads(json_string)
            # print("The RMSE for validate is: " + json_dict["homo_secureboost_0"]["validate"]["root_mean_squared_error"])

            # save train pipeline
            # filename = "pipeline_homo_sbt_saved_" + str(iteration) + "1100-24jul" + ".pkl"
            # pipeline.dump(filename)

            print("End training for iteration: " + str(iteration))

            ###########
            # predict
            ###########

            print("Reached prediction")
            # pipeline = PipeLine.load_model_from_file(filename)
            pipeline.deploy_component([pipeline.datatransform_0, pipeline.homo_secureboost_0])  # deploy so that it can be used in predict stage

            reader_1 = Reader(name='reader_1')
            datatransform_1 = DataTransform(name='datatransform_1')

            reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=party_C_test_data)
            reader_1.get_party_instance(role='host', party_id=host[0]).component_param(table=party_B_test_data)
            reader_1.get_party_instance(role='host', party_id=host[1]).component_param(table=party_A_test_data)

            datatransform_1.get_party_instance(role='guest', party_id=guest).component_param(with_label=True,
                                                                                             output_format="dense",
                                                                                             label_type="float")
            datatransform_1.get_party_instance(role='host', party_id=host[0]).component_param(with_label=True,
                                                                                              output_format="dense",
                                                                                              label_type="float")
            datatransform_1.get_party_instance(role='host', party_id=host[1]).component_param(with_label=True,
                                                                                              output_format="dense",
                                                                                              label_type="float")


            predict_pipeline = PipeLine()  # new pipeline object
            predict_pipeline.add_component(reader_1)
            # data is {training data : prediction data}
            predict_pipeline.add_component(pipeline,
                                           data=Data(predict_input={pipeline.datatransform_0.input.data: reader_1.output.data}))

            # define evaluation component
            evaluation_1 = Evaluation(name="evaluation_1")
            evaluation_1.get_party_instance(role="guest", party_id=guest).component_param(need_run=True, eval_type="regression")
            evaluation_1.get_party_instance(role="host", party_id=host[0]).component_param(need_run=True, eval_type="regression")
            evaluation_1.get_party_instance(role="host", party_id=host[1]).component_param(need_run=True, eval_type="regression")
            predict_pipeline.add_component(evaluation_1, data=Data(data=pipeline.homo_secureboost_0.output.data))

            # run predict model
            print("Start prediction process")
            predict_pipeline.predict(job_parameters)
            print(f"Predict Evaluation summary:\n{json.dumps(predict_pipeline.get_component('evaluation_1').get_summary(), indent=4)}")
            # json_string = json.dumps(pipeline.get_component('evaluation_1').get_summary())
            # json_dict = json.loads(json_string)
            # print("The RMSE for validate is: " + json_dict["homo_secureboost_0"]["validate"]["root_mean_squared_error"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
