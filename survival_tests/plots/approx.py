import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
t = "0.458707 &  0.431291 &  0.422140 &  0.393256 &  0.384608 &  0.394363 &  0.402074 &  0.392175 &  0.381926 &  0.377657 &  0.379779 &  0.376703 &  0.388309 &  0.401986 &  0.412185"
y = np.zeros(len(x))

t_split = t.split("&")
for i, c in enumerate(t_split):
    val = c.replace(" ", "")
    y[i] = val

x_2 = np.square(x)

matrix = np.array([x_2, x, np.ones(len(y))])
matrix = np.asmatrix(matrix)

matrix_y = np.asmatrix(y).transpose()

matrix_1 = matrix * matrix_y
matrix_2 = matrix * matrix.transpose()
x = np.linalg.solve(matrix_2, matrix_1)

print(x)

