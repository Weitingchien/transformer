import math

z = [5, 7, 4, 2, 1, 6, 3, 8, 9]


def softmax(z):
    exp_z = [math.exp(zi) for zi in z]
    sum_exp_z = sum(exp_z)
    softmax_output = [zi / sum_exp_z for zi in exp_z]
    return softmax_output


def test():
    softmax_output = softmax(z)
    print(softmax_output)
    print(sum(softmax_output))


if __name__ == "__main__":
    test()
