import time
import os

def get_time():
    time_ = time.strftime('%Y-%m-%d-%H-%M-%S')
    return str(time_)

def mkdirs(path):
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except:
        return False