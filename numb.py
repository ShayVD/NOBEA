from math import *
import numpy as np

def multi_michalewicz(x):
    """
    Michalewicz function for chromosomes with 2 or more genes
    domain = [0, pi]
    :param x: [] array of floats
    :return: float
    """
    j = [i for i in range(1, len(x)+1)]
    sum = 0
    for i in range(len(x)):
        sum += sin(x[i]) * sin(j[i] * x[i] ** 2 / pi) ** 20
    return - sum

def egg_holder(x):
    n = len(x)
    sum = 0
    for i in range(n-1):
        sum += (x[i+1] + 47) * sin(sqrt(abs(x[i+1] + 47 + x[i]/2))) + x[i] * sin(sqrt(abs(x[i] - (x[i+1] + 47))))
    return - sum

def sine_envelope(x):
    sum = 0
    n = len(x)
    for i in range(n-2):
        xi = x[i]
        nxtx = x[i+1]
        sum += 0.5 + sin(sqrt(nxtx ** 2 + xi ** 2) - 0.5) ** 2 / (0.001 * (nxtx ** 2 + xi ** 2) + 1) **2
    return - sum

def sphere(x):
    """
    domain = [-5, 5]
    :param x:
    :return:
    """
    sum = 0
    n = len(x)
    for i in range(n):
        sum += x[i] ** 2
    return sum