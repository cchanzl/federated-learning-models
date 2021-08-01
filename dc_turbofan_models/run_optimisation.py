import pyautogui as pg
import time
import numpy as np
import random

print(pg.size())
x_coor = [100, 1600, 100, 1600]
y_coor = [100, 100, 800, 800]

base = 'python FATE-Ubuntu/dc_turbofan_models/turbofan_fed_avg_worker.py --server-host-ip 127.0.1.1 --server-port 8080 '

ITERATIONS = 100
for i in range(ITERATIONS):
    count = 0

    # randomised grid search
    lr_list = list(np.arange(5, 20 + 1, 0.5) / 100)
    epoch_list = list(np.arange(10, 50 + 1, 5))
    nodes_list = [[8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [128, 256, 512], [256, 512, 1024]]
    batch_size_list = list(np.arange(10, 50 + 1, 5))
    iter_list = list(np.arange(5, 15 + 1, 1))
    dropout_list = list(np.arange(5, 50 + 1, 0.5) / 10)

    # init parameters
    lr = random.sample(lr_list, 1)[0]
    nodes_per_layer = random.sample(nodes_list, 1)[0]
    batch_sz = random.sample(batch_size_list, 1)[0]
    iter_no = random.sample(iter_list, 1)[0]
    dropout = random.sample(dropout_list, 1)[0]

    batch_size = '--batch-size ' + str(batch_sz) + ' '
    learn_rate = '--learn-rate ' + str(lr) + ' '
    itera_roun = '--iter-rounds ' + str(iter_no) + ' '
    layer_one = '--layer-one ' + str(nodes_per_layer[0]) + ' '
    layer_two = '--layer-two ' + str(nodes_per_layer[1]) + ' '
    layer_three = '--layer-three ' + str(nodes_per_layer[2]) + ' '
    drop_out = '--drop-out ' + str(dropout) + ' '
    party_code = '--party-code '

    print(f"Iteration: {i+1}"
          f" Learning rate: {lr}"
          f" Batch size: {batch_sz}"
          f" Iteration round: {iter_no}"
          f" Layer one: {nodes_per_layer[0]}"
          f" Layer two: {nodes_per_layer[1]}"
          f" Layer three: {nodes_per_layer[2]}"
          f" Drop out: {dropout}")

    for x, y in zip(x_coor, y_coor):
        time.sleep(3)
        server = "python FATE-Ubuntu/dc_turbofan_models/turbofan_fed_avg_server.py " + \
                 batch_size + learn_rate + itera_roun + layer_one + layer_two + layer_three + drop_out

        if x == 100 and y == 100:
            pg.moveTo(x, y, duration=1)  # x, y
            pg.click(x, y)
            pg.typewrite(server)
            pg.typewrite(["enter"])
            continue

        party = chr(65+count)
        worker = base + batch_size + learn_rate + itera_roun + layer_one + layer_two + layer_three + \
                 drop_out + party_code + party

        pg.moveTo(x, y, duration=1)  # x, y
        pg.click(x, y)
        pg.typewrite(worker)
        pg.typewrite(["enter"])
        count += 1

    pg.moveTo(x_coor[0], y_coor[0], duration=1)  # x, y
    pg.click(x_coor[0], y_coor[0])
    time.sleep(iter_no*5)
    pg.hotkey("ctrlleft", "c")
