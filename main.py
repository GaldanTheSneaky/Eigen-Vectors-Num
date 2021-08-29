import numpy as np
from eigen import get_eigenvalues
from eigen import get_eigenvectors
from eigen import get_frobenius


def k_values(values, eps=0.1):
    k = []
    for i in range(len(values)):
        k.append(1)
    for i in range(len(values)):
        for j in range(len(values)):
            if abs(values[i] - values[j]) <= eps and k[i] != 0:
                k[i] += 1

    mids =[]
    for j in range(len(values)):
        max = 0
        mid = -1
        for i in range(len(values)):
            if k[i] > max and i not in mids:
                max = k[i]
                mid = i
        mids.append(mid)
        if max != 0:
            for i in range(len(values)):
                if abs(values[mid] - values[i]) <= eps:
                    if mid != i:
                        k[i] = 0
            k[mid] -= 1

    return k

def k_vectors(vectors, eps=0.1):
    k = []
    for i in range(len(vectors)):
        k.append(1)
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            eq_counter = 0
            for l in range(len(vectors[0])):
                if abs(vectors[i][l] - vectors[j][l]) <= eps and k[i] != 0:
                    eq_counter += 1
            if eq_counter == len(vectors[0]):
                k[i] += 1

    mids = []
    for j in range(len(vectors)):
        max = 0
        mid = -1
        for i in range(len(vectors)):
            if k[i] > max and i not in mids:
                max = k[i]
                mid = i
        mids.append(mid)
        if max != 0:
            for i in range(len(vectors)):
                eq_counter = 0
                for l in range(len(vectors[0])):
                    if abs(vectors[mid][l] - vectors[i][l]) <= eps:
                        eq_counter += 1
                    if eq_counter == len(vectors[0]) and mid != i:
                        k[i] = 0
            k[mid] -= 1

    return k


file = open('input.txt', 'r')

task = int(file.readline())
order = int(file.readline())
A = np.loadtxt('input.txt', skiprows=2, usecols=range(order), max_rows=order, delimiter=" ")

open('output.txt', 'w').close()
output_file = open("output.txt", "a")

output_file.write(str(get_frobenius(A)))
output_file.write("\n")
output_file.write("\n")
eigenvalues = get_eigenvalues(A)

if task == 1:
    k = k_values(eigenvalues)
    for i in range(len(eigenvalues)):
        if k[i] != 0:
            output_file.write(str(eigenvalues[i]))
            output_file.write("\n")
            output_file.write(str(np.linalg.det(np.subtract(A, np.identity(order) * eigenvalues[i]))))
            output_file.write("\n")
            output_file.write(str(k[i]))
            output_file.write("\n")
else:
    eigenvectors = get_eigenvectors(A)
    k = k_vectors(eigenvectors)
    for i in range(len(eigenvectors)):
        if k[i] != 0:
            output_file.write(str(eigenvectors[i]))
            output_file.write("\n")
            output_file.write(str((np.matmul(A, eigenvectors[i])) - (eigenvalues[i] * eigenvectors[i])))
            output_file.write("\n")
            output_file.write(str(k[i]))
            output_file.write("\n")

