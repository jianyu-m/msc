
import numpy
import random
import math


# directed graph
def generate(n, p):
    num = 0
    g = numpy.zeros((n, n))
    if p >= 1:
        return None
    rand_max = math.ceil(1 / p)
    for i in range(n):
        for j in range(n):
            r = random.randint(0, rand_max - 1)
            if r == 0:
                g[i][j] = 1
                num += 1
    print(num)
    return g


if __name__ == "__main__":
    A = generate(1000, 0.01)
    print(A)