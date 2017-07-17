
import numpy


def file_to_matrix(filename, n):
    matrix = numpy.zeros((n, n), numpy.float64)
    with open(filename, 'r') as f:
        line_m = f.read()
        lines = line_m.split("\n")
        for line in lines:
            if line.startswith("#"):
                continue
            line_arr = line.strip().split(" ")
            x, y, z = int(line_arr[0]), int(line_arr[1]), int(line_arr[2])
            matrix[x - 1][y - 1] = z
    return matrix