import numpy as np

arr = np.array([
    [1,3,23],
    [34,66,77],
    [43,12,67]])
print(arr[0,0] ,arr[1,1],arr[2,2])
print(arr[-1])
print(arr[0:3])
print(arr[:,1])
print(arr[0:2])
print(arr[:,0:2])

print(arr[1:,:2])