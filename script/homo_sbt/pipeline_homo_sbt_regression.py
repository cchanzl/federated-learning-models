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

    # set job initiator
    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    # 0 for train data, 1 for test data
    datatransform_0, datatransform_1 = DataTransform(name="datatransform_0"), DataTransform(name='datatransform_1')
    reader_0, reader_1 = Reader(name="reader_0"), Reader(name='reader_1')

    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=party_A_train_data)
    reader_0.get_party_instance(role='host', party_id=host[0]).component_param(table=party_B_train_data)
    reader_0.get_party_instance(role='host', party_id=host[1]).component_param(table=party_C_train_data)

    datatransform_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True,
                                                                              output_format="dense",
                                                                              label_type="float")
    datatransform_0.get_party_instance(role='host', party_id=host[0]).component_param(with_label=True,
                                                                               output_format="dense",
                                                                               label_type="float")
    datatransform_0.get_party_instance(role='host', party_id=host[1]).component_param(with_label=True,
                                                                               output_format="dense",
                                                                               label_type="float")

    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=party_A_test_data)
    reader_1.get_party_instance(role='host', party_id=host[0]).component_param(table=party_B_test_data)
    reader_1.get_party_instance(role='host', party_id=host[1]).component_param(table=party_C_test_data)

    datatransform_1.get_party_instance(role='guest', party_id=guest).component_param(with_label=True,
                                                                              output_format="dense",
                                                                              label_type="float")
    datatransform_1.get_party_instance(role='host', party_id=host[0]).component_param(with_label=True,
                                                                               output_format="dense",
                                                                               label_type="float")
    datatransform_1.get_party_instance(role='host', party_id=host[1]).component_param(with_label=True,
                                                                               output_format="dense",
                                                                               label_type="float")

    homo_secureboost_0 = HomoSecureBoost(name="homo_secureboost_0",
                                         num_trees=30,  # 20 is best
                                         task_type='regression',
                                         # None,'cross_entropy','lse','lae','log_cosh','tweedie','fair','huber'
                                         objective_param={"objective": "lse"},
                                         tree_param={"max_depth": 10},
                                         validation_freqs=3)

    job_parameters = JobParameters(backend=backend, work_mode=work_mode)

    print("Start training")
    train = True
    if train:
        evaluation_0 = Evaluation(name='evaluation_0', eval_type='regression')
        pipeline.add_component(reader_0)
        pipeline.add_component(datatransform_0, data=Data(data=reader_0.output.data))
        pipeline.add_component(reader_1)

        # pipeline.add_component(datatransform_1, data=Data(data=reader_1.output.data), model=Model(datatransform_0.output.model))

        # https://github.com/FederatedAI/FATE/tree/178f04d1a58181359d6550b4673d4b4dc72a778f/python/fate_client/pipeline/component
        homo_data_split_1 = HomoDataSplit(name='homo_data_split_1')
        pipeline.add_component(homo_data_split_1, data=Data(data=datatransform_0.output.data))

        pipeline.add_component(homo_secureboost_0, data=Data(train_data=homo_data_split_1.output.data.train_data,
                                                             validate_data=homo_data_split_1.output.data.validate_data))
        pipeline.add_component(evaluation_0, data=Data(homo_secureboost_0.output.data))
        pipeline.compile()

        pipeline.fit(job_parameters)

        # save train pipeline
        pipeline.dump("pipeline_homo_sbt_saved.pkl")

    print("End training")

    ###########
    # predict
    ###########

    print("Reached prediction")
    pipeline = PipeLine.load_model_from_file('pipeline_homo_sbt_saved.pkl')
    pipeline.deploy_component([datatransform_0, homo_secureboost_0])  # deploy so that it can be used in predict stage

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
