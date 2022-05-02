"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""
import numpy as np
from numpy.random import uniform

import assignment1


MAXITER = 1000



class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        n=30
        points=np.linspace(a, b, n)
        ass1 = assignment1.Assignment1()
        canHandleArray=True
        try:
            arr=np.array([1,2,3])
            f(arr)
        except:
            canHandleArray=False
        def get_avg_y(f, x):
            toReturn = np.empty(shape=10000)
            toReturn[0]=f(x)
            for i in range(0, 10000):
                toReturn[i] = f(x)
            return np.float32(np.average(toReturn))
        def get_avg_y_Arr(f, x, numofpieces):
            ToReturn=np.empty(shape=[10000,numofpieces])
            for i in range(0, 10000):
                for j in range(numofpieces):
                    ToReturn[i][j] = f(x[j])
            return np.float32(np.average(ToReturn, axis=0))
        avg_arr = []
        if canHandleArray:
            ass1.x_array = points
            temp=get_avg_y_Arr(f, points, n)
            for i in range(n):
                avg_arr.append(np.array([points[i], temp[i]]))
            ass1.arr = ass1.BezierArray(avg_arr)
            g = ass1.forword
        else:
            for i in range(n):
                avg_arr.append(np.array([points[i], get_avg_y(f, points[i])]))
            ass1.x_array = points
            ass1.arr = ass1.BezierArray(avg_arr)
            g = ass1.forword
        return g
import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEquals(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEquals(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1
        print(mse)

    def test_err_2(self):
        f = poly(1, 2, 3, 4, 1, 2, 3, 4)
        df = DELAYED(0.01)(f)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=-1, b=5, d=50, maxtime=20)
        # print(ff)
        T = time.time() - T
        print("done in ", T)
        mse = 0
        for x in uniform(low=-2, high=2, size=1):
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1
        print(mse)


if __name__ == "__main__":
    unittest.main()
