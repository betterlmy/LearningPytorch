# GPU性能测试
import time
import GPUtil
import torch


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.tik = None
        self.times = []
        self.state = 'stopped'

    def start(self):
        """启动计时器"""
        if self.state == 'running':
            print("timer is still running")
        else:
            self.tik = time.time()
            self.state = 'running'

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        if self.state == 'running':
            self.times.append(time.time() - self.tik)
            self.state = 'stopped'
            return self.times[-1]
        else:
            print(self.state)
            print("timer is not running")
            return None

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)


def test_once(device, times=10, width=40000, height=3000):
    """测试一次"""
    timer = Timer()
    for x in range(times):
        timer.start()
        a = torch.randn(width, height, requires_grad=False, device=device)
        b = torch.randn(height, width, requires_grad=False, device=device)
        if 'cuda' in device.type:
            torch.cuda.empty_cache()
        c = torch.matmul(a, b)
        del a, b, c
        if x == 0 or (x + 1) % 5 == 0:
            print(f"第{x + 1}次运算结束")
        timer.stop()

    return timer.sum(), timer.avg()


def main():
    devices = []
    name = []
    for gpu in GPUtil.getGPUs():
        if gpu.memoryUtil < 0.3:
            """仅挑选GPU使用率小于30%"""
            name.append(gpu.name)
            devices.append(torch.device(f'cuda:{gpu.id}'))
    devices.append(torch.device('cpu'))
    name.append('cpu')

    for device in devices:
        if 'cpu' in device.type:
            """CPU测试时间较长,只测5次"""
            print(f"当前测试的是{name[devices.index(device)]}")
            sum, avg = test_once(device, 5)
            print(f"总耗时{sum:.2f}s,平均{avg:.2f}s")
        elif torch.cuda.is_available():
            print(f"当前测试的是{name[devices.index(device)]}:{str(device.index)}")
            sum, avg = test_once(device, 20)
            print(f"总耗时{sum:.2f}s,平均{avg:.2f}s")
        else:
            print('没有可用的GPU,跳过测试')


if __name__ == '__main__':
    main()
