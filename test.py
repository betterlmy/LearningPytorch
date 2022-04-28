def checkType(x):
    # 以RxCx的形式表示
    if x[0] == 'R' and 'C' in x:
        if 48 <= ord(x[x.rindex('C')-1]) <= 57:
            return 'RC'
    return 'XY'


def convert26(x):
    dict = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13,
        'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25,
        'Z': 26
    }
    # 将字母转换为数字
    if type(x) == type(1):
        result = ''
        while x % 26 != 0:
            result += chr(x % 26 + 64)
            x = x//26
        return result[::-1]

    if type(x) == type("a"):
        result = 0
        for i in range(len(x)):
            result += dict[x[i]] * (26 ** (len(x) - i - 1))
        return result


def convertRC2XY(x):
    c_index = x.rfind('C')
    X1 = x[1:c_index]
    X2 = int(x[c_index + 1:])
    # print(X1,X2)
    # print(convert26(X2), end='')
    # print(X1)
    print(f'{convert26(X2)}{X1}')


def convertXY2RC(x):
    split = 0
    for i in range(len(x)):
        if 48 <= ord(x[i]) <= 57:
            split = i
            break
    X1 = x[:split]
    X2 = x[split:]
    # print("R", end='')
    # print(X2, end='')
    # print("C", end='')
    # print(convert26(X1))
    print(f"R{X2}C{convert26(X1)}")


def main():
    # times = int(input())
    # inputs = []
    # for _ in range(times):
    #     inputs.append(input().upper())
    # for x in inputs:
    #     if checkType(x) == 'RC':
    #         # 如果是单元格坐标
    #         convertRC2XY(x)
    #     else:
    #         # 如果是系统坐标
    #         convertXY2RC(x)

    times = int(input())
    for _ in range(times):
        x=input().upper()
        if checkType(x) == 'RC':
            # 如果是单元格坐标
            convertRC2XY(x)
        else:
            # 如果是系统坐标
            convertXY2RC(x)


if __name__ == '__main__':
    main()
