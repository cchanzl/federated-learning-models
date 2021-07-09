#  python /main/script/pipeline-upload.py --base /main/data
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

from pipeline.backend.config import Backend, WorkMode
from pipeline.backend.pipeline import PipeLine

# path to data
# default fate installation path
DATA_BASE = "/data/projects/fate"

# site-package ver
# import site
# DATA_BASE = site.getsitepackages()[0]


def main(data_base=DATA_BASE):
    breast = False  # Set to true if want to use breast dataset
    party_A_train = "party_A_x_train.csv"
    party_B_train = "party_B_x_train.csv"
    party_C_train = "party_C_x_train.csv"

    party_A_train_name = "nasa_A"
    party_B_train_name = "nasa_B"
    party_C_train_name = "nasa_C"

    party_A_test = "party_A_x_test.csv"
    party_B_test = "party_B_x_test.csv"
    party_C_test = "party_C_x_test.csv"

    party_A_test_name = "nasa_A_test"
    party_B_test_name = "nasa_B_test"
    party_C_test_name = "nasa_C_test"

    if breast:
        party_A_train = "breast_homo_guest.csv"
        party_B_train = "breast_homo_host.csv"
        party_A_train_name = "breast_homo_guest"
        party_B_train_name = "breast_homo_host"
        # parties config

    # does not matter what this number is. only used for upload. not relevant to model training.
    guest = 1234

    # 0 for eggroll, 1 for spark
    backend = Backend.EGGROLL
    # 0 for standalone, 1 for cluster
    work_mode = WorkMode.STANDALONE
    # use the work mode below for cluster deployment
    # work_mode = WorkMode.CLUSTER

    # partition for data storage
    partition = 4

    # table name and namespace, used in FATE job configuration
    party_A_train_dict = {"name": party_A_train_name, "namespace": f"experiment"}
    party_B_train_dict = {"name": party_B_train_name, "namespace": f"experiment"}
    party_C_train_dict = {"name": party_C_train_name, "namespace": f"experiment"}

    party_A_test_dict = {"name": party_A_test_name, "namespace": f"experiment"}
    party_B_test_dict = {"name": party_B_test_name, "namespace": f"experiment"}
    party_C_test_dict = {"name": party_C_test_name, "namespace": f"experiment"}

    # https://fate.readthedocs.io/en/latest/_build_temp/python/fate_client/pipeline/README.html
    pipeline_upload = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest)
    # add upload data info
    # path to csv file(s) to be uploaded, modify to upload designated data
    # This is an example for standalone version.
    # For cluster version, you will need to upload your data on each party respectively.

    pipeline_upload.add_upload_data(file=os.path.join(data_base, party_A_train),
                                    table_name=party_A_train_dict["name"],             # table name
                                    namespace=party_A_train_dict["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, party_B_train),
                                    table_name=party_B_train_dict["name"],
                                    namespace=party_B_train_dict["namespace"],
                                    head=1, partition=partition)

    pipeline_upload.add_upload_data(file=os.path.join(data_base, party_C_train),
                                    table_name=party_C_train_dict["name"],
                                    namespace=party_C_train_dict["namespace"],
                                    head=1, partition=partition)

    pipeline_upload.add_upload_data(file=os.path.join(data_base, party_A_test),
                                    table_name=party_A_test_dict["name"],
                                    namespace=party_A_test_dict["namespace"],
                                    head=1, partition=partition)

    pipeline_upload.add_upload_data(file=os.path.join(data_base, party_B_test),
                                    table_name=party_B_test_dict["name"],
                                    namespace=party_B_test_dict["namespace"],
                                    head=1, partition=partition)

    pipeline_upload.add_upload_data(file=os.path.join(data_base, party_C_test),
                                    table_name=party_C_test_dict["name"],
                                    namespace=party_C_test_dict["namespace"],
                                    head=1, partition=partition)

    # upload data
    pipeline_upload.upload(work_mode=work_mode, backend=backend, drop=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--base", "-b", type=str,
                        help="data base, path to directory that contains examples/data")

    args = parser.parse_args()
    if args.base is not None:
        main(args.base)
    else:
        main()
