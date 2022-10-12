import time


def test():
    print(2)


def get_now(name: str = None) -> float:
    print(f'hello,{name}')
    test()
    return time.time()


if __name__ == "__main__":
    get_now()
