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
    Sphere Model
    Domain: -5.12 -> 5.12 (-100 -> 100)
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
    Schwefel's Problem 2.22
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
    Schwefel's Problem 1.2
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
    Schwefel's Problem 2.21
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
    Generalised Rosenbrock's Function
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
    Step Function
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
    Quartic Function i.e. Noise
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
    Generalised Schwefel's Problem 2.26
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
    Generalised Rastrigin's Function
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
    Ackley's Function
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
    Generalised Griewank Function
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
    Generalised Penalised Function
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
    Generalised Penalised Function
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


a_matrix = [[-32, -16, 0, 16, 32] * 5, [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5]


def f14(x):
    """
    M. Shekel's Foxholes Function
    Dimensions: 2
    Domain: -65.54 -> 65.54
    Minimum value: f(31.95) = 0.998 (~1)
    :param x:
    :return:
    """
    sum = 0
    for j in range(24):
        secondsum = 0
        for i in range(1):
            secondsum += (x[i] + a_matrix[i][j])**6
        sum += (j + 1 + secondsum)**-1
    total = ((1/500) + sum)**-1
    return total


a_vector = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
b_vector = [1/0.25, 1/0.5, 1, 1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 1/16]


def f15(x):
    """
    N. Kowalik's Function
    Dimensions: 4
    Domain: -5 -> 5
    Minimum value: f(0.19,0.19,0.12,0.14) = 0.0003075
    :param x:
    :return:
    """
    sum = 0
    for i in range(10):
        sum += (a_vector[i]-((x[0]*((b_vector[i]**2)+b_vector[i]*x[1]))/((b_vector[i]**2)+b_vector[i]*x[2]+x[3])))**2
    return sum


def f16(x):
    """
    O. Six-Hump Camel-Back Function
    Dimensions: 2
    Domain: -5 -> 5
    Minimum value: f(-0.09,0.71) = -1.0316
    :param x:
    :return:
    """
    return 4*x[0]**2 - 2.1*x[0]**4 + (1/3)*x[0]**6 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4


def f17(x):
    """
    P. Branin Function
    Dimensions: 2
    Domain: -5 -> 15
    Minimum value: f(9.42,2.47) = 0.398
    :param x:
    :return:
    """
    return (x[1] - (5.1/4*(pi**2))*(x[0]**2) + (5/pi)*x[0] - 6)**2 + 10*(1 - (1/(8*pi)))*cos(x[0])+10


def f18(x):
    """
    Q. Goldstein-Price Function
    Dimensions: 2
    Domain: -2 -> 2
    Minimum value: f(0,-1) = 3
    :param x:
    :return:
    """
    return ((1+((x[0]+x[1]+1)**2)*(19-(14*x[0])+((3*x[0])**2)-(14*x[1])+(6*x[0]*x[1])+((3*x[1])**2)))*(30+(((2*x[0]) -
            (3*x[1]))**2)*(18-(32*x[0])+(12*(x[0]**2))+(48*x[1])-(36*x[0]*x[1])+(27*(x[1]**2)))))


a_hartman19 = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
c_hartman = [1, 1.2, 3, 3.2]
p_hartman19 = [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470], [0.1091, 0.8732, 0.5547], [0.038150, 0.5743, 0.8828]]


def f19(x):
    """
    R. Hartman's Family
    Dimensions: 3
    Domain: 0 -> 1
    Minimum value: f(0.114,0.556,0.852) = -3.86
    :param x:
    :return:
    """
    sum = 0
    for i in range(4):
        secondsum = 0
        for j in range(3):
            secondsum += a_hartman19[i][j]*((x[j]-p_hartman19[i][j])**2)
        sum += c_hartman[i]*exp(-secondsum)
    return -sum


a_hartman20 = [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
p_hartman20 = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]


def f20(x):
    """
    R. Hartman's Family
    Dimensions: 6
    Domain: 0 -> 1
    Minimum value: f(0.201,0.150,0.477,0.275,0.311,0.657) = -3.32
    :param x:
    :return:
    """
    sum = 0
    for i in range(4):
        secondsum = 0
        for j in range(3):
            secondsum += a_hartman20[i][j]*((x[j]-p_hartman20[i][j])**2)
        sum += c_hartman[i]*exp(-secondsum)
    return -sum


a_shekel = [[4]*4, [1]*4, [8]*4, [6]*4, [3, 7]*2, [2, 9]*2, [5, 5, 3, 3], [8, 1]*2, [6, 2]*2, [7, 3.6]*2]
c_shekel = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]


def dot_product(vector_1, vector_2):
    return sum(x*y for x, y in zip(vector_1, vector_2))


def subtract_vectors(vector_1, vector_2):
    return [x - y for (x, y) in zip(vector_1, vector_2)]


def f21(x):
    """
    S. Shekel's Family
    Dimensions: 4
    Domain: 0 -> 10
    Minimum value: f(~4) = -10.2
    :param x:
    :return:
    """
    sum = 0
    for i in range(4):
        vector_1 = subtract_vectors(x, a_shekel[i])
        vector_2 = subtract_vectors(x, a_shekel[i])
        sum += (dot_product(vector_1, vector_2) + c_shekel[i])**-1
    return -sum


def f22(x):
    """
   S. Shekel's Family
   Dimensions: 4
   Domain: 0 -> 10
   Minimum value: f(~4) = -10.4
   :param x:
   :return:
   """
    sum = 0
    for i in range(6):
        vector_1 = subtract_vectors(x, a_shekel[i])
        vector_2 = subtract_vectors(x, a_shekel[i])
        sum += (dot_product(vector_1, vector_2) + c_shekel[i]) ** -1
    return -sum


def f23(x):
    """
   S. Shekel's Family
   Dimensions: 4
   Domain: 0 -> 10
   Minimum value: f(~4) = -10.5
   :param x:
   :return:
   """
    sum = 0
    for i in range(9):
        vector_1 = subtract_vectors(x, a_shekel[i])
        vector_2 = subtract_vectors(x, a_shekel[i])
        sum += (dot_product(vector_1, vector_2) + c_shekel[i]) ** -1
    return -sum
