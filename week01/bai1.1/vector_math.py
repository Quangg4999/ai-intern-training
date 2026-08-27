import math
import numpy as np

A = [1, 2, 3, 4]
B = [2, 4, 6, 8]
C = [8, 1, 0, 2]

# tich vo huong
def dot_product(a, b): 
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# do dai vector
def vector_norm(a):    
    return math.sqrt(dot_product(a, a))

# khoang cach euclide , 2 vector cach nhau bao nhiêu
def euclidean_distance(a, b): 
    hieu = []
    for i in range(len(a)):
        hieu.append(a[i] - b[i])
    return vector_norm(hieu)

# Do tuong dong cosine , 2 vector cung huong hay khong
def cosine_similarity(a, b):
    tu_so = dot_product(a, b)
    mau_so = vector_norm(a) * vector_norm(b)
    if mau_so == 0:
        return 0.0
    return tu_so / mau_so

print("distance(A, B):",  euclidean_distance(A, B))
print("distance(A, C):",  euclidean_distance(A, C))
print("cos(A, B):", cosine_similarity(A, B))
print("cos(A, C):", cosine_similarity(A, C))

# Kiem chung bang numpy
A_np = np.array(A)
B_np = np.array(B)
C_np = np.array(C)
print("distance(A, B) by numpy:", np.linalg.norm(A_np - B_np))
print("distance(A, C) by numpy:", np.linalg.norm(A_np - C_np))
print("cos(A, B) by numpy:", np.dot(A_np, B_np) / (np.linalg.norm(A_np) * np.linalg.norm(B_np)))
print("cos(A, C) by numpy:", np.dot(A_np, C_np) / (np.linalg.norm(A_np) * np.linalg.norm(C_np)))

print(" Sai so distance(A,B):", abs(euclidean_distance(A, B) - np.linalg.norm(A_np - B_np)))
print(" Sai so distance(A,C):", abs(euclidean_distance(A, C) - np.linalg.norm(A_np - C_np)))
print(" Sai so cos(A,B):", abs(cosine_similarity(A, B) - (np.dot(A_np, B_np) / (np.linalg.norm(A_np) * np.linalg.norm(B_np)))))
print(" Sai so cos(A,C):", abs(cosine_similarity(A, C) - (np.dot(A_np, C_np) / (np.linalg.norm(A_np) * np.linalg.norm(C_np)))))
