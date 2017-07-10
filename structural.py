from queue import Queue

import numpy


def dfs(u, matching, check, g, n):
    for v in range(n):
        if g[u][v] == 0 or check[v]:
            continue
        check[v] = True
        if matching[v] == -1 or dfs(matching[v], matching, check, g, n):
            matching[u] = v
            matching[v] = u
            return True
    return False


def reset(arr):
    for i in range(len(arr)):
        arr[i] = False


def process_graph(g):
    n_r, n_c = g.shape
    n = n_r

    ans = 0
    matching = [-1 for i in range(n)]
    check = [False for i in range(n)]
    for u in range(n):
        if matching[u] == -1:
            reset(check)
            if dfs(u, matching, check, g, n):
                ans += 1
    print("find " + str(ans) + " matches")
    return matching


def process_graph_bfs(g):
    n = 6
    ans = 0
    matching = [-1 for i in range(n)]
    check = [-1 for i in range(n)]
    prev = [-1 for i in range(n)]
    q = []
    for i in range(n):
        if matching[i] == -1:
            q.clear()
            q = q + [i]
            prev[i] = -1
            flag = False
            while len(q) != 0 and not flag:
                u = q[0]
                for v in range(n):
                    if g[u][v] == 0 or check[v] == i:
                        continue
                    else:
                        check[v] = i
                        q = q + [matching[v]]
                        if matching[v] >= 0:
                            prev[matching[v]] = u
                        else:
                            flag = True
                            d = u
                            e = v
                            while d != -1:
                                t = matching[d]
                                matching[d] = e
                                matching[e] = d
                                d = prev[d]
                                e = t
                q = q[1:]
            if matching[i] != -1:
                ans += 1
    print(ans)


if __name__ == "__main__":
    A = numpy.array([(0, 1, 1, 1, 1, 1),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 1),
                     (1, 0, 0, 0, 1, 0)], numpy.float16)

    A = numpy.array([(0, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 0),
                     (1, 0, 0, 0, 0, 1),
                     (1, 0, 0, 0, 0, 0)], numpy.int16)

    # input is a transmission matrix representing the graph
    B = process_graph(A)
    # for l in un_lam:
    #     print(sympy.Matrix(l * numpy.identity(N) - A))

    print(B)