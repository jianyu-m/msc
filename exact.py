
import numpy
from numpy.linalg import matrix_rank


# return max geo-multiplicity and its value
def max_e(arr, unique_arr, n):
    max_multi = -1
    max_lambda = 0
    for i, a in enumerate(unique_arr):
        r = n - matrix_rank(a * numpy.identity(n) - arr)
        if r > max_multi:
            max_multi = r
            max_lambda = a
    return max_lambda, max_multi


def get_none_zero(arr, to):
    for i, val in enumerate(arr):
        if val != 0 and i >= to:
            return i, val
    return -1, 0


def element_column(arr, n):
    for i in range(n):
        k, p = get_none_zero(arr[i], i)
        if k == -1:
            continue
        else:
            if k != i:
                arr[:, k], arr[:, i] = arr[:, i], arr[:, k].copy()
            if k != 1:
                arr[:, i] = arr[:, i] / arr[i][i]
            for j in range(i + 1, n):
                arr[:, j] = arr[:, j] - arr[i][j] * arr[:, i]
    return arr


def construct_v(arr, n):
    base = matrix_rank(arr[0])
    l = []
    if base == 0:
        l.append(0)
    for i in range(1, len(arr)):
        cal_rank = matrix_rank(arr[0:i + 1, :])
        if cal_rank > base:
            base = cal_rank
        else:
            l.append(i)

    z = numpy.zeros((n, len(l)))
    for i, li in enumerate(l):
        z[li][i] = 1
    return z


def process_graph(graph):
    n_r, n_c = graph.shape
    if not n_r == n_c:
        print("wrong matrix")
        return
    n = n_r
    lamda, v = numpy.linalg.eig(graph)
    unique_lamda = numpy.unique(lamda)
    lamda_gm, lamda_gm_c = max_e(graph, unique_lamda, n)
    B = construct_v(element_column(graph - lamda_gm * numpy.identity(n), n), n)

    # verify this result
    _, b_c = B.shape
    r = matrix_rank(numpy.concatenate(
        (lamda_gm * numpy.identity(n), numpy.zeros((n, b_c))), axis=1) -
                    numpy.concatenate((A, B), axis=1))
    print(lamda_gm, lamda_gm_c, r)
    assert r == n
    return B


if __name__ == "__main__":
    A = numpy.array([(0, 1, 1, 1, 1, 1),
                     (1, 1, 0, 0, 0, 0),
                     (1, 0, 1, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 1),
                     (1, 0, 0, 0, 1, 0)], numpy.float32)

    A = numpy.array([(0, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 1),
                     (1, 0, 0, 0, 0, 0)], numpy.float32)

    # input is a transmission matrix representing the graph
    B = process_graph(A)

    # for l in un_lam:
    #     print(sympy.Matrix(l * numpy.identity(N) - A))

    print(B)
