#  python /main/script/pipeline-upload.py -config /main/script/homo_sbt/config_5.yaml
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

import os
import argparse
from pipeline.utils.tools import load_job_config
from pipeline.backend.config import Backend, WorkMode
from pipeline.backend.pipeline import PipeLine

# path to data
# default fate installation path
DATA_BASE = "/data/projects/fate"

# site-package ver
# import site
# DATA_BASE = site.getsitepackages()[0]


def main(config="../../config.yaml", data_base='/main/data'):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)

    # update role details
    parties = config.parties
    guest = parties.guest[0]
    # host_0 = parties.host[0]
    # host_1 = parties.host[1]
    arbiter = parties.arbiter[0]

    # key parameters
    num_parties = 5
    balanced = False
    train_file = {}
    train_name = {}
    test_file = {}
    test_name = {}
    host_list = []

    for i in range(num_parties):
        ending = ".csv"
        if balanced:
            ending = "_balanced.csv"

        identifier = str(i+1)
        if num_parties == 3:
            identifier = chr(65 + i)

        train_file["party_{0}".format(i)] = "party_" + identifier + "_train" + ending
        train_name["party_{0}".format(i)] = "nasa_" + identifier

        test_file["party_{0}".format(i)] = "party_" + identifier + "_test" + ending
        test_name["party_{0}".format(i)] = "nasa_" + identifier + "_test"

        if i != num_parties-1:
            host_list.append(parties.host[i])

    # 0 for eggroll, 1 for spark
    backend = Backend.EGGROLL

    # 0 for standalone, 1 for cluster
    work_mode = WorkMode.STANDALONE

    # use the work mode below for cluster deployment
    # work_mode = WorkMode.CLUSTER

    # partition for data storage
    partition = 4

    # https://github.com/FederatedAI/FATE/blob/178f04d1a58181359d6550b4673d4b4dc72a778f/examples/pipeline/homo_sbt/pipeline-homo-sbt-binary-multi-host.py
    # https://fate.readthedocs.io/en/latest/_build_temp/python/fate_client/pipeline/README.html
    pipeline_upload = PipeLine().set_initiator(role="guest", party_id=guest)
    pipeline_upload = pipeline_upload.set_roles(guest=guest, host=host_list, arbiter=arbiter)
    # add upload data info
    # path to csv file(s) to be uploaded, modify to upload designated data
    # This is an example for standalone version.
    # For cluster version, you will need to upload your data on each party respectively.

    # upload train data
    for i in range(num_parties):
        pipeline_upload.add_upload_data(file=os.path.join(data_base, train_file['party_' + str(i)]),
                                        table_name=train_name['party_' + str(i)],             # table name
                                        namespace=f"experiment",                                # namespace
                                        head=1, partition=partition)                            # data info

        pipeline_upload.add_upload_data(file=os.path.join(data_base, test_file['party_' + str(i)]),
                                        table_name=test_name['party_' + str(i)],
                                        namespace=f"experiment",
                                        head=1, partition=partition)

    # upload data
    pipeline_upload.upload(work_mode=work_mode, backend=backend, drop=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
