# !/usr/bin/env python
# coding=utf-8
# author=yexiaozhu
import numpy as np
from numpy.matlib import randn

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
# print arr1
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
# print arr2
# print arr2.ndim
# print arr2.shape
# print arr1.dtype
# print arr2.dtype
# print np.zeros(10)
# print np.zeros((3, 6))
# print np.empty((2, 3, 2))
# print np.arange(15)
arr3 = np.array([1, 2, 3], dtype=np.float64)
arr4 = np.array([1, 2, 3], dtype=np.int32)
# print arr3.dtype
# print arr4.dtype
arr = np.array([1, 2, 3, 4, 5])
# print arr.dtype
float_arr = arr.astype(np.float64)
# print float_arr.dtype
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
# print arr
# print arr.astype(np.int32)
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
# print numeric_strings.astype(np.float64)
int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
# print int_array.astype(calibers.dtype)
empty_uint32 = np.empty(8, dtype='u4')
# print empty_uint32
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# print arr
# print arr * arr
# print 1 / arr
# print arr - arr
# print arr ** 0.5
arr = np.arange(10)
# print arr
# print arr[5]
# print arr[5:8]
# arr[5:8] = 12
# print arr
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print arr2d[2]
# print arr2d[0][2]
# print arr2d[0, 2]
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print arr3d
# print arr3d[0]
old_values = arr3d[0].copy()
arr3d[0] = 42
# print arr3d
arr3d[0] = old_values
# print arr3d
# print arr3d[1, 0]
# print arr[1:6]
# print arr2d
# print arr2d[:2]
# print arr2d[:2, 1:]
# print arr2d[1, :2]
# print arr2d[2, :1]
# print arr2d[:, :1]
# print arr2d[:2, 1:]
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = randn(7, 4)
# print names
# print data
# print names == 'Bob'
# print data[names == 'Bob']
# print data[names == 'Bob', 2:]
# print data[names == 'Bob', 3]
# print names != 'Bob'
# print data[-(names == 'Bob')]
mask = (names == 'Bob') | (names == 'Will')
# print mask
# print data[mask]
data[data < 0] = 0
# print data
data[names != 'Joe'] = 7
# print data
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
# print arr
# print arr[[4, 3, 0, 6]]
# print arr[[-3, -5, -7]]
arr = np.arange(32).reshape((8, 4))
# print arr
# print arr[[1, 5, 7, 2], [0, 3, 1, 2]]
# print arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
# print arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]
arr = np.arange(15).reshape((3, 5))
# print arr
# print arr.T
arr = np.random.randn(6, 3)
# print np.dot(arr.T, arr)
arr = np.arange(16).reshape((2, 2, 4))
# print arr
# print arr.transpose((1, 0, 2))
# print arr.swapaxes(1, 2)
arr = np.arange(10)
# print np.sqrt(arr)
# print np.exp(arr)
x = randn(8)
y = randn(8)
# print x, y
# print np.maximum(x, y)
arr = randn(7) * 5
# print np.modf(arr)
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
# print ys
import matplotlib.pyplot as plt
z = np.sqrt(xs ** 2 + ys ** 2)
# print z
# plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
# plt.title("Image plot of $\sqrt{x^2 + y^2} for a grid of values")
# plt.show()
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
# print result
# print np.where(cond, xarr, yarr)
arr = randn(4, 4)
# print arr
# print np.where(arr > 0, 2, -2)
# print np.where(arr > 0, 2, arr)
arr = np.random.randn(5, 4)
# print arr.mean()
# print np.mean(arr)
# print arr.sum()
# print arr.mean(axis=1)
# print arr.sum(0)
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# print arr.cumsum(0)
# print arr.cumprod(1)
arr = randn(100)
# print (arr > 0).sum()
# print (arr < 0).sum()
bools = np.array([False, False, True, False])
# print bools.any()
# print bools.all()
arr = randn(8)
# print arr
# arr.sort()
# print arr
arr = randn(5, 3)
# print arr
# arr.sort(1)
# print arr
large_arr = randn(1000)
large_arr.sort()
# print large_arr[int(0.05 * len(large_arr))]# 5%分位数
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# print np.unique(names)
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
# print np.unique(ints)
# print sorted(set(names))
values = np.array([6, 0, 0, 3, 2, 5, 6])
# print np.in1d(values, [2, 3, 6])
arr = np.arange(10)
np.save('some_array', arr)
# print np.load('some_array.npy')
np.savez('array_archive.npz', a=arr, b=arr)
# arch = np.load('array_archive.npz')
# print arch['b']
arr = np.loadtxt('array_ex.txt', delimiter=',')
# print arr
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8., 9.]])
# print x, y
# print x.dot(y)
# print np.dot(x, np.ones(3))
from numpy.linalg import inv, qr
X = randn(5, 5)
mat = X.T.dot(X)
# print inv(mat)
# print mat.dot(inv(mat))
q, r = qr(mat)
# print r
samples = np.random.normal(size=(4, 4))
# print samples
from random import normalvariate
import datetime
import time
starttime = datetime.datetime.now()
start = time.clock()
N = 1000000
# samples = [normalvariate(0, 1) for _ in xrange(N)]
np.random.normal(size=N)
endtime = datetime.datetime.now()
end = time.clock()
# print endtime - starttime
# print '时间：%f s' %(end - start)
import random
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
# print np.random.seed(12345)
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
# print walk.min()
# print walk.max()
# print (np.abs(walk) >= 10).argmax()
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
# print walks
# print walks.max()
# print walks.min()
hits30 = (np.abs(walks) >= 30).any(1)
# print hits30
# print hits30.sum() # Number that hit 30 or -30
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
print crossing_times.mean()
steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))
# print steps
print steps