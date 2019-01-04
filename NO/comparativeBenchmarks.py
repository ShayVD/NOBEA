"""
Xin  Yao, Senior Member, IEEE,
Yong Liu, Student Member, IEEE,
and Guangming Lin
Evolutionary Programming Made Faster
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 3, NO. 2, JULY 1999
"""
from math import *
import random


def f1(x):
    """
    Domain: -5.12 -> 5.12
    Minimum value: f(0) = 0
    :param x:
    :return:
    """
    sum = 0
    for i in range(len(x)):
        sum += x[i]**2
    return sum


def f2(x):
    """
    Domain: -10 -> 10
    Minimum value: f(0) = 0
    :param x:
    :return:
    """
    sum = 0
    prod = 0
    for i in range(len(x)):
        sum += abs(x[i])
        prod *= x[i]
    total = sum + prod
    return total


def f3(x):
    """
    Domain: -100 -> 100
    Minimum value: f(0) = 0
    :param x:
    :return:
    """
    sum = 0
    for i in range(len(x)):
        secondsum = 0
        for j in range(i):
            secondsum += x[i]
        sum += secondsum**2
    return sum


def f4(x):
    """
    Domain: -100 -> 100
    Minimum value: f(0) = 0
    :param x:
    :return:
    """
    max = None
    for i in range(len(x)):
        if max is None or abs(x[i]) > max:
            max = abs(x[i])
    return max


def f5(x):
    """
    Domain: -30 -> 30
    Minimum value: f(1) = 0
    :param x:
    :return:
    """
    sum = 0
    for i in range(len(x)-1):
        sum += 100*(x[i+1] - (x[i]**2))**2 + (x[i] - 1)**2
    return sum


def f6(x):
    """
    Domain: -100 -> 100
    Minimum value: f(p) = 0, -0.5 <= p <= 0.5
    :param x:
    :return:
    """
    sum = 0
    for i in range(len(x)):
        sum += (abs(x[i] + 0.5))**2
    return sum


def f7(x):
    """
    Domain: -1.28 -> 1.28
    Minimum value: f(0) = 0
    :param x:
    :return:
    """
    sum = 0
    for i in range(len(x)):
        sum += (i + 1) * x[i]**4
    sum += random.uniform(0, 1)
    return sum


def f8(x):
    """
    Domain: -500 -> 500
    Minimum value: f(420.97) = 12569.5/41898.3
    :param x:
    :return:
    """
    sum = 0
    for i in range(len(x)):
        sum += -x[i] * sin(sqrt(abs(x[i])))
    return sum


def f9(x):
    """
    Domain: -5.12 -> 5.12
    Minimum value: f(0) = 0
    :param x:
    :return:
    """
    sum = 0
    for i in range(len(x)):
        sum += (x[i]**2) - 10*cos(2*pi*x[i]) + 10
    return sum


def f10(x):
    """
    Domain: -32 -> 32
    Minimum value: f(0) = 0
    :param x:
    :return:
    """
    sum = 0
    n = len(x)
    for i in range(n):
        a = 0
        b = 0
        for i in range(n):
            a += x[i]**2
            b += cos(2*pi*x[i])
        sum += -20*exp(-0.2*sqrt((1/n)*a)) - exp((1/n)*b) + 20 + e
    return sum


def f11(x):
    """
    Domain: -600 -> 600
    Minimum value: f(0) = 0
    :param x:
    :return:
    """
    sum = 0
    prod = 0
    for i in range(len(x)):
        sum += x[i]**2
        prod *= cos(x[i]/(sqrt(i+1)))
    total = (1/4000)*sum + prod + 1
    return total


def y(xi):
    return 1 + 0.25*(xi + 1)


def u(x, a, b, c):
    if x > a:
        return b*(x-a)**c
    elif -a <= x <= a:
        return 0
    else:
        return b*(-x-a)**c


def f12(x):
    """
    Domain: -50 -> 50
    Minimum value: f(-1) = 0
    :param x:
    :return:
    """
    n = len(x)
    sum = 0
    secondsum = 0
    for i in range(n):
        if i < n - 1:
            sum += (((y(x[i]) - 1)**2) * (1 + 10*(sin(pi*y(x[i+1])))**2)) + (y(x[n-1])-1)**2
        secondsum += u(x[i], 10, 100, 4)
    total = (pi/n)*((10*sin(pi*y(x[1])))**2 + sum) + secondsum
    return total


def f13(x):
    """
    TODO CHECK AGAIN
    Domain: -50 -> 50
    Minimum value: f(1,...,1,-4.76) = -1.1428
    :param x:
    :return:
    """
    n = len(x)
    sum = 0
    secondsum = 0
    for i in range(n):
        if i < n - 1:
            sum += (((x[i] - 1)**2) * (1 + (sin(3*pi*x[i+1]))**2)) + (x[n-1] - 1)*(1 + (sin(2*pi*x[n-1]))**2)
        secondsum += u(x[i], 5, 100, 4)
    total = 0.1*((sin(3*pi*x[1]))**2 + sum) + secondsum
    return total


def f14(x):
    sum = 0
    for i in range(24):
        pass
    total = ((1/500) + sum)**-1