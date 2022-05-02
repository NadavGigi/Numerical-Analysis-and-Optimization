"""
In this assignment you should find the intersection points for two functions.
"""



import numpy as np
import time
import random
from collections.abc import Iterable
MAX_ITER = 10000


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def regulaFalsi(self, f, RangeA, RangeB, boundary):
        temporary = RangeA
        if (abs(f(RangeA))) <= boundary:
            return RangeA
        if (abs(f(RangeB))) <= boundary:
            return RangeB
        if f(RangeA) * f(RangeB) >= 0:
            return
        for i in range(MAX_ITER):
            temporary = (RangeA * f(RangeB) - RangeB * f(RangeA)) / (f(RangeB) - f(RangeA))
            if abs(f(temporary)) <= boundary:
                return temporary
            elif f(temporary) * f(RangeA) < 0:
                RangeB = temporary
            else:
                RangeA = temporary

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        X = []
        new_func = lambda x: f1(x) - f2(x)
        points = int(abs(b - a) * 12)
        arr = np.linspace(a, b, points)
        for i in range(len(arr) - 1):
            res = self.regulaFalsi(new_func, arr[i], arr[i + 1] + maxerr, maxerr)
            if len(X)>0 and res is not None and abs(res-X[len(X)-1])<0.001:
                continue
            else:
                X += [res] if res is not None else []

        return X


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        print(X)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
