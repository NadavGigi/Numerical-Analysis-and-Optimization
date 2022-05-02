"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random



class Assignment1:
    def __init__(self):
        self.x_array = []
        self.arr = []
        pass


    def forword(self, x):
        arr_x = self.x_array
        arr_t = self.arr
        min = 0
        for i in range(0, len(arr_x)):
            if min <= x < arr_x[i]:
                func = lambda t: arr_t[i - 1](t)[0] - x
                flag = True
                a = 0
                b = 0
                c = 1
                boundary = 0.001
                while flag:
                    b = a - (c - a) * func(a) / (func(c) - func(a))
                    if func(a) * func(b) < 0:
                        c = b
                    else:
                        a = b
                    flag = abs(func(b)) > boundary
                t = b
                return arr_t[i - 1](t)[1]
            min = arr_x[i]
        return arr_t[len(arr_t) - 1](x)[1]

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        self.x_array = np.linspace(a, b, n)
        points = []
        for x in self.x_array:
            points.append(np.array([x, f(x)]))
        self.arr = self.BezierArray(points)

        g = self.forword
        return g

    def get_cubic(self,ai, bi, ci, di):
        return lambda t: np.power(1 - t, 3) * ai + 3 * np.power(1 - t, 2) * t * bi + 3 * (1 - t) * np.power(t,
                                                                                                            2) * ci + np.power(
            t, 3) * di

    def BezierArray(self, points):
        n = len(points) - 1
        TEMP1 = 4 * np.identity(n)
        np.fill_diagonal(TEMP1[1:], 1)
        np.fill_diagonal(TEMP1[:, 1:], 1)
        TEMP1[0, 0] = 2
        TEMP1[n - 1, n - 1] = 7
        TEMP1[n - 1, n - 2] = 2

        TEMP2 = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
        TEMP2[0] = points[0] + 2 * points[1]
        TEMP2[n - 1] = 8 * points[n - 1] + points[n]

        TEMP2 = np.array(TEMP2)
        ARR1 = self.ThomasAlgorithem(TEMP1, TEMP2)  # TDM solver

        ARR2 = [0] * n
        for i in range(n - 1):
            ARR2[i] = 2 * points[i + 1] - ARR1[i + 1]
        ARR2[n - 1] = (ARR1[n - 1] + points[n]) / 2
        return [
            self.get_cubic(points[i], ARR1[i], ARR2[i], points[i + 1])
            for i in range(len(points) - 1)
        ]

    def ThomasAlgorithem(self, M, argMetrix):
        #Initilizing metrixes
        Metrix1=[M[i][i - 1] for i in range(1, len(M[0]))]

        Metrix2=[M[i][i] for i in range(len(M[0]))]

        Metrix3=[M[i][i + 1] for i in range(len(M[0]) - 1)]

        nf = len(argMetrix)
        for it in range(1, nf):
            temporary = Metrix1[it - 1] / Metrix2[it - 1]
            Metrix2[it] = Metrix2[it] - temporary * Metrix3[it - 1]
            argMetrix[it] = argMetrix[it] - temporary * argMetrix[it - 1]
        tempMetrix = Metrix2
        tempMetrix[-1] = argMetrix[-1] / Metrix2[-1]
        for il in range(nf - 2, -1, -1):
            tempMetrix[il] = (argMetrix[il] - Metrix3[il] * tempMetrix[il + 1]) / Metrix2[il]

        return tempMetrix




##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("{} is {}".format("time", T))
        print("{} is {}".format("error", mean_err))

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
