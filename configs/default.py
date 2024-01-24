import numpy as np
import torch
import pycuda.driver as cuda
from collections import OrderedDict

configs_dict = OrderedDict()

# Device
configs_dict["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda.init()
for i in range(cuda.Device.count()):
    device = cuda.Device(i)
    print(f"Device {i}: {device.name()}")
    thread_num = device.max_threads_per_block
    print(f"Maximum Threads per Block: {thread_num}")
    props = device.get_attributes()
    block_num = props.get(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    print(f"Total Blocks: {block_num}")
# 请根据上面输出的thread_num和block_num自行决定
configs_dict["cuda_threads"] = 512
configs_dict["cuda_blocks"] = 40

# Train
configs_dict["epochs"] = int(1e+4)
configs_dict["early_stop"] = 10
configs_dict["eval_interval"] = 12

# Env
configs_dict["call_strikes"] = [100.0]
configs_dict["put_strikes"] = [100.0]
configs_dict["use_stock"] = True
configs_dict["trans_cost"] = True


configs_dict['action_scaler'] = 2

## Sim Eng
# SDE
configs_dict["mu"] = 0.
configs_dict["kappa"] = 3
configs_dict["theta"] = 0.25
configs_dict["xi"] = 0.3
configs_dict["mean"] = np.array([0, 0])
configs_dict["correlation"] = 0.6

# Time
configs_dict["T"] = 252
configs_dict["iteration_steps"] = 1500

# init
configs_dict["x0"] = 100
configs_dict["v0"] = 0.25

# Delta
configs_dict["shock_percent"] = 0.05
configs_dict["rf"] = 0.012

# FIA state p
configs_dict["prob_exit_yearunit"] = 0.04
configs_dict["prob_death_yearunit"] = 0.01

configs_dict['episode_length'] = 1500


## SAC model
configs_dict['random_action_steps'] = 500000 ## steps to take random action

configs_dict['learning_rate'] = 0.0001
configs_dict['tau'] = 0.005
configs_dict['gamma'] = 0.99
configs_dict['alpha'] = 0.2
configs_dict['alpha_tune'] = True

configs_dict['batch_size'] = 1024
configs_dict['update_interval'] = 25
configs_dict['eval_epochs'] = 15

## Buffer
configs_dict['max_buff_size'] = 5000000
configs_dict['save_location'] = '/buffer'

