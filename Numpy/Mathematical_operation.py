import numpy as np

a= np.array([[12,45,66],[12,45,67],[90,200,455]])
b= np.array([[18,5,96],[1,1,1],[78,23,44]])

print(a+b)
print(a*b)

a= np.array([[10,20],[30,40]])
b= np.array([[12,14],[32,34]])

print(a@b)
print(np.sqrt(a))