
import numpy
from numpy.linalg import matrix_rank

import sparse_reader

global eps

# calculate geo-multiplicity
def lambda_geo(arr, unique_arr, n):
    geo = [0 for i in range(len(unique_arr))]
    for i, a in enumerate(unique_arr):
        geo[i] = n - matrix_rank(a * numpy.identity(n) - arr)
    return geo


def get_none_zero(arr, to):
    for i, val in enumerate(arr):
        if val > eps and i >= to:
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


def set_v(arr, v):
    base = matrix_rank(arr[0])
    if base == 0:
        v[0] = True
    for i in range(1, len(arr)):
        cal_rank = matrix_rank(arr[0:i + 1, :])
        if cal_rank > base:
            base = cal_rank
        else:
            v[i] = True


def process_graph(graph):
    global eps
    n_r, n_c = graph.shape
    if not n_r == n_c:
        print("wrong matrix")
        return
    n = n_r
    eps = numpy.finfo(graph.dtype).eps * n
    lamda, v = numpy.linalg.eig(graph)
    unique_lamda = numpy.unique(lamda)
    lambda_arr = lambda_geo(graph, unique_lamda, n)

    controller = max(lambda_arr)

    msc_upper = sum(lambda_arr)

    # find msc
    v = [False for i in range(n)]
    for lam in unique_lamda:
        # union the v set
        set_v(element_column(graph - lam * numpy.identity(n), n), v)
        element_column(graph - lam * numpy.identity(n), n)
    # calculate the final result
    msc = 0
    for vi in v:
        if vi:
            msc += 1
    # controller number | msc | upper_bound | unique_lambda | upper - lower
    return controller, msc, min(msc_upper, n), len(unique_lamda) - 1, min(msc_upper, n) - controller

if __name__ == "__main__":
    A = numpy.array([(0, 1, 1, 1, 1, 1),
                     (1, 1, 0, 0, 0, 0),
                     (1, 0, 1, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 1),
                     (1, 0, 0, 0, 1, 0)], numpy.float64)

    A = numpy.array([(0, 0, 0, 0, 0, 0),
                     (0, 1, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 1),
                     (1, 0, 0, 0, 0, 0)], numpy.float64)

    # input is a transmission matrix representing the graph
    # msc = process_graph(sparse_reader.file_to_matrix("celegans_metabolic[503].net", 503))
    # msc = process_graph(sparse_reader.file_to_matrix("dataset/Electronic circuits-s208a[122].txt", 122))
    # msc = process_graph(sparse_reader.file_to_matrix("dataset/Electronic circuits-s420a[252].txt", 252))
    # msc = process_graph(sparse_reader.file_to_matrix("dataset/Electronic circuits-s838a[515].txt", 515))
    # msc = process_graph(sparse_reader.file_to_matrix("dataset/Organizational-Consulting[46].txt", 46))
    # msc = process_graph(sparse_reader.file_to_matrix("dataset/Organizational-Freeman[46].txt", 46))
    # msc = process_graph(sparse_reader.file_to_matrix("dataset/Trust-prison inmate[67].txt", 67))
    # msc = process_graph(A)
    print("controller number | msc | upper_bound | unique_lambda - 1 <= upper - lower")
    print(msc)
    # for l in un_lam:
    #     print(sympy.Matrix(l * numpy.identity(N) - A))
