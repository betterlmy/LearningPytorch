from viztracer import VizTracer


def f(x):
    x = 2 * x
    print(x)


def main():
    f(1)


with VizTracer():
    main()
