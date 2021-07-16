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
import argparse
import json
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.data_transform import DataTransform
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.utils.tools import load_job_config
from pipeline.runtime.entity import JobParameters
from pipeline.component import Evaluation
from pipeline.interface.model import Model


def run_homo_nn_pipeline(config, namespace, nn_component, num_host):
    if isinstance(config, str):
        config = load_job_config(config)

    # read input from config.yaml
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host  # multiple other parties
    arbiter = parties.arbiter[0]

    party_A_train_data = {"name": "nasa_A", "namespace": f"experiment"}
    party_A_test_data = {"name": "nasa_A_test", "namespace": f"experiment"}

    party_B_train_data = {"name": "nasa_B", "namespace": f"experiment"}
    party_B_test_data = {"name": "nasa_B_test", "namespace": f"experiment"}

    party_C_train_data = {"name": "nasa_C", "namespace": f"experiment"}
    party_C_test_data = {"name": "nasa_C_test", "namespace": f"experiment"}
    
    # guest_train_data = data["guest"]
    # host_train_data = data["host"][:num_host]
    # for d in [guest_train_data, *host_train_data]:
    #     d["namespace"] = f"{d['namespace']}{namespace}"
    #
    # hosts = config.parties.host[:num_host]

    # set job initiator
    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    # 0 for train data, 1 for test data
    dataio_0, dataio_1 = DataIO(name="dataio_0"), DataIO(name='dataio_1')
    reader_0, reader_1 = Reader(name="reader_0"), Reader(name='reader_1')

    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=party_A_train_data)
    reader_0.get_party_instance(role='host', party_id=host[0]).component_param(table=party_B_train_data)
    reader_0.get_party_instance(role='host', party_id=host[1]).component_param(table=party_C_train_data)

    dataio_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True,
                                                                              output_format="dense",
                                                                              label_type="float")
    dataio_0.get_party_instance(role='host', party_id=host[0]).component_param(with_label=True,
                                                                               output_format="dense",
                                                                               label_type="float")
    dataio_0.get_party_instance(role='host', party_id=host[1]).component_param(with_label=True,
                                                                               output_format="dense",
                                                                               label_type="float")

    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=party_A_test_data)
    reader_1.get_party_instance(role='host', party_id=host[0]).component_param(table=party_B_test_data)
    reader_1.get_party_instance(role='host', party_id=host[1]).component_param(table=party_C_test_data)

    dataio_1.get_party_instance(role='guest', party_id=guest).component_param(with_label=True,
                                                                              output_format="dense",
                                                                              label_type="float")
    dataio_1.get_party_instance(role='host', party_id=host[0]).component_param(with_label=True,
                                                                               output_format="dense",
                                                                               label_type="float")
    dataio_1.get_party_instance(role='host', party_id=host[1]).component_param(with_label=True,
                                                                               output_format="dense",
                                                                               label_type="float")

    # Set up pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    # pipeline.add_component(reader_1)
    # pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data), model=Model(dataio_0.output.model))
    pipeline.add_component(nn_component, data=Data(train_data=dataio_0.output.data))
    # evaluation_0 = Evaluation(name="evaluation_0", eval_type="regression")
    # pipeline.add_component(evaluation_0, data=Data(data=nn_component.output.data))

    # Compile and fit pipeline
    pipeline.compile()
    job_parameters = JobParameters(backend=config.backend, work_mode=config.work_mode)
    pipeline.fit(job_parameters)

    # print(pipeline.get_component("homo_nn_0").get_summary())
    # #print(f"Evaluation summary:\n{json.dumps(pipeline.get_component('evaluation_0').get_summary(), indent=4)}")
    pipeline.deploy_component([dataio_0, nn_component])

    # predict
    predict_pipeline = PipeLine()  # new pipeline object
    predict_pipeline.add_component(reader_1)
    predict_pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data))
    predict_pipeline.add_component(pipeline,
                                   data=Data(predict_input={pipeline.dataio_0.input.data: reader_0.output.data}),)
    # run predict model
    predict_pipeline.predict(job_parameters)


def runner(main_func):
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str, help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main_func(args.config)
    else:
        main_func()
