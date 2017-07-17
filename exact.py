
import numpy
from numpy.linalg import matrix_rank

import sparse_reader


# calculate geo-multiplicity
def lambda_geo(arr, unique_arr, n):
    geo = [0 for i in range(len(unique_arr))]
    for i, a in enumerate(unique_arr):
        geo[i] = n - matrix_rank(a * numpy.identity(n) - arr)
    return geo


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
        cal_rank = matrix_rank(arr[0:i + 1, :], eps)
        if cal_rank > base:
            base = cal_rank
        else:
            l.append(i)

    z = numpy.zeros((n, len(l)))
    for i, li in enumerate(l):
        z[li][i] = 1
    return z


def v_count(arr):
    base = matrix_rank(arr[0])
    num = 0
    if base == 0:
        num += 1
    for i in range(1, len(arr)):
        cal_rank = matrix_rank(arr[0:i + 1, :])
        if cal_rank > base:
            base = cal_rank
        else:
            num += 1
    return num


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


def compare_independent(arr, row):
    independent = True
    for arr_row in arr:
        if max(arr_row - row) == eps:
            independent = False
            break
    return independent


def process_graph(graph):
    n_r, n_c = graph.shape
    if not n_r == n_c:
        print("wrong matrix")
        return
    n = n_r
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
    return controller, msc, msc_upper

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
    # msc = process_graph(sparse_reader.file_to_matrix("dataset/Organizational-Consulting.txt", 46))
    msc = process_graph(sparse_reader.file_to_matrix("dataset/Trust-prison inmate.txt", 67))
    # msc = process_graph(A)
    print(msc)
    # for l in un_lam:
    #     print(sympy.Matrix(l * numpy.identity(N) - A))
