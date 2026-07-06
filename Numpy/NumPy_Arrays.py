import numpy as np

# 1D Array 
# marks = np.array([23,44,56,78])
# print(marks)

# 2D Array
# arr = np.array([[23,44,56,78] , [23,56,76,34]])
# print(arr)

# 3D Array
# arr = np.array([[23,44,56,78] , [23,56,76,34] , [47,90,15,67]])
# print(arr)

# Using dtype
# arr = np.array([1,2,3], dtype=float)

# print(arr)


# Common Array Creation Functions

#zeros()
# arr= np.zeros(5)
# print(arr)


# arr = np.zeros((3,3,3))
# print(arr)


# ones()
# arr = np.ones((2,2))
# print(arr)

#full()
# arr = np.full((2,2),6)
# print(arr)

#eye()
# arr = np.eye(3)
# print(arr)

#arange 
# arr = np.arange(10,100,5)
# print(arr)


# linespace
# arr = np.linspace(0,100 , 5 , dtype=int)
# print(arr)

#Array Attributes
arr = np.array([[1,4,34,3,2],[43,42,67,89,45]])
print(arr)
print(arr.ndim)
print(arr.shape)
print(arr.size)
print(arr.dtype)
print(arr.itemsize)
print(arr.nbytes)