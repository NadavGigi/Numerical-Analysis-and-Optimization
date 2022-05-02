"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""
import math

import numpy as np

from functionUtils import AbstractShape


class MyShape(AbstractShape):
    def __init__(self):
        pass


class Assignment5:
    def __init__(self):
        self.origin=0
        self.refvec = [0, 1]

    def clockwiseangle_and_distance(self,point):
        vector = [point[0] - self.origin[0], point[1] - self.origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * self.refvec[0] + normalized[1] * self.refvec[1]  
        diffprod = self.refvec[1] * normalized[0] - self.refvec[0] * normalized[1]
        angle = math.atan2(diffprod, dotprod)

        if angle < 0:
            return 2 * math.pi + angle, lenvector

        return angle, lenvector
    def calculate(self, contour: callable, n):
        area = 0
        points = contour(n)
        points = points.T
        x_array = points[0]
        y_array = points[1]
        j = n - 1
        for i in range(n):
            area = area + ((x_array[j] + x_array[i]) * (y_array[j] - y_array[i]))
            j = i
        return area / 2

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        area = 0
        n = 10
        areacurr = self.calculate(contour, n)
        while n*2 <= 5000 and abs(area - areacurr) >= maxerr:
            n = n * 2
            area = areacurr
            areacurr = self.calculate(contour, n)
        return np.float32(areacurr)
    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:

        ArrayOfSamples=np.array([sample(), ])
        for i in range(10000):
            ArrayOfSamples=np.append(ArrayOfSamples,[np.array(sample())],axis=0)
        self.origin=[2,3]
        ArrayOfSamples = np.array(sorted(ArrayOfSamples,key=self.clockwiseangle_and_distance))
        ArrayOfPointsOnFigure =ArrayOfSamples

        class MyShape(AbstractShape):
            def __init__(self, ARR):
                self.ARR = ARR
                super(MyShape, self)
            def area(self):
                x=ArrayOfPointsOnFigure[:, 0]
                y=ArrayOfPointsOnFigure[:, 1]
                npoints=len(ArrayOfPointsOnFigure)
                area = 0
                j = npoints - 1
                for i in range((npoints)):
                    area += (x[j] + x[i]) * (y[j] - y[i])
                    j = i
                return area / 2

            def contour(self, n: int):
                pass

            def sample(self):
                pass

        return MyShape(ArrayOfPointsOnFigure)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()