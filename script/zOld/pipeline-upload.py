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
    breast = False
    guest_file = "party_A_x_train.csv"
    host_file = "party_B_x_train.csv"
    guest_name = "nasa_A_guest"
    host_name = "nasa_B_host"
    if breast:
        guest_file = "breast_homo_guest.csv"
        host_file = "breast_homo_host.csv"
        guest_name = "breast_homo_guest"
        host_name = "breast_homo_host"
        # parties config
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
    dense_data = {"name": guest_name, "namespace": f"experiment"}
    tag_data = {"name": host_name, "namespace": f"experiment"}

    pipeline_upload = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest)
    # add upload data info
    # path to csv file(s) to be uploaded, modify to upload designated data
    # This is an example for standalone version. For cluster version, you will need to upload your data
    # on each party respectively.


    pipeline_upload.add_upload_data(file=os.path.join(data_base, guest_file),
                                    table_name=dense_data["name"],             # table name
                                    namespace=dense_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, host_file),
                                    table_name=tag_data["name"],
                                    namespace=tag_data["namespace"],
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
