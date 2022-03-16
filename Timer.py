import time

import numpy as np


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        # 每次暂停后计数
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return self.sum() / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
