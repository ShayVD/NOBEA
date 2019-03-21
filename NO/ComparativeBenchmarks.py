"""
Xin  Yao, Senior Member, IEEE,
Yong Liu, Student Member, IEEE,
and Guangming Lin
Evolutionary Programming Made Faster
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 3, NO. 2, JULY 1999
"""
from NO.Benchmark import Benchmark
from numpy import random
import math


class ComparativeBenchmarks(object):

    a_matrix = [[-32, -16, 0, 16, 32] * 5, [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5]

    a_vector = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]

    b_vector = [1/0.25, 1/0.5, 1, 1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 1/16]

    a_hartman19 = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]

    c_hartman = [1, 1.2, 3, 3.2]

    p_hartman19 = [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470], [0.1091, 0.8732, 0.5547],
                   [0.038150, 0.5743, 0.8828]]

    a_hartman20 = [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]]

    p_hartman20 = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]

    a_shekel = [[4] * 4, [1] * 4, [8] * 4, [6] * 4, [3, 7] * 2, [2, 9] * 2, [5, 5, 3, 3], [8, 1] * 2, [6, 2] * 2,
                [7, 3.6] * 2]

    c_shekel = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]

    @staticmethod
    def y(xi):
        return 1 + 0.25 * (xi + 1)

    @staticmethod
    def u(x, a, b, c):
        if x > a:
            return b * (x - a) ** c
        elif -a <= x <= a:
            return 0
        else:
            return b * (-x - a) ** c

    @staticmethod
    def dot_product(vector_1, vector_2):
        return sum(x * y for x, y in zip(vector_1, vector_2))

    @staticmethod
    def subtract_vectors(vector_1, vector_2):
        return [x - y for (x, y) in zip(vector_1, vector_2)]

    @staticmethod
    def benchmarks():
        return [ComparativeBenchmarks.f1(), ComparativeBenchmarks.f2(), ComparativeBenchmarks.f3(),
                ComparativeBenchmarks.f4(), ComparativeBenchmarks.f5(), ComparativeBenchmarks.f6(),
                ComparativeBenchmarks.f7(), ComparativeBenchmarks.f8(), ComparativeBenchmarks.f9(),
                ComparativeBenchmarks.f10(), ComparativeBenchmarks.f11(), ComparativeBenchmarks.f12(),
                ComparativeBenchmarks.f13(), ComparativeBenchmarks.f14(), ComparativeBenchmarks.f15(),
                ComparativeBenchmarks.f16(), ComparativeBenchmarks.f17(), ComparativeBenchmarks.f18(),
                ComparativeBenchmarks.f19(), ComparativeBenchmarks.f20(), ComparativeBenchmarks.f21(),
                ComparativeBenchmarks.f22(), ComparativeBenchmarks.f23()]

    @staticmethod
    def benchmarks_any_dimension():
        return [ComparativeBenchmarks.f1(), ComparativeBenchmarks.f2(), ComparativeBenchmarks.f3(),
                ComparativeBenchmarks.f4(), ComparativeBenchmarks.f5(), ComparativeBenchmarks.f6(),
                ComparativeBenchmarks.f7(), ComparativeBenchmarks.f8(), ComparativeBenchmarks.f9(),
                ComparativeBenchmarks.f10(), ComparativeBenchmarks.f11(), ComparativeBenchmarks.f12(),
                ComparativeBenchmarks.f13()]

    @staticmethod
    def benchmarks_specific_dimension():
        return [ComparativeBenchmarks.f14(), ComparativeBenchmarks.f15(), ComparativeBenchmarks.f16(),
                ComparativeBenchmarks.f17(), ComparativeBenchmarks.f18(), ComparativeBenchmarks.f19(),
                ComparativeBenchmarks.f20(), ComparativeBenchmarks.f21(), ComparativeBenchmarks.f22(),
                ComparativeBenchmarks.f23()]

    @staticmethod
    def benchmarks_single_objective_optimisation():
        return [ComparativeBenchmarks.f1(), ]

    @staticmethod
    def f1(domain=(-5.12, 5.12), dimensions=30):
        """
        Sphere Model
        Domain: -5.12 -> 5.12
        Minimum Value: f(0) = 0

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(len(x)):
                sum += x[i]**2
            return sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0, name="Sphere Model")

    @staticmethod
    def f2(domain=(-10, 10), dimensions=30):
        """
        Schwefel's Problem 2.22
        Domain: -10 -> 10
        Minimum value: f(0) = 0

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            prod = 0
            for i in range(len(x)):
                sum += abs(x[i])
                prod *= abs(x[i])
            total = sum + prod
            return total
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0,
                         name="Schwefel's Problem 2.22")

    @staticmethod
    def f3(domain=(-100, 100), dimensions=30):
        """
        Schwefel's Problem 1.2
        Domain: -100 -> 100
        Minimum value: f(0) = 0

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(len(x)):
                secondsum = 0
                for j in range(i):
                    secondsum += x[j]
                sum += secondsum**2
            return sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0,
                         name="Schwefel's Problem 1.2")

    @staticmethod
    def f4(domain=(-100, 100), dimensions=30):
        """
        Domain: -100 -> 100
        Minimum value: f(0) = 0

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            max = None
            for i in range(len(x)):
                if max is None or abs(x[i]) > max:
                    max = abs(x[i])
            return max
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0,
                         name="Schwefel's Problem 2.21")

    @staticmethod
    def f5(domain=(-30, 30), dimensions=30):
        """
        Generalised Rosenbrock's Function
        Domain: -30 -> 30
        Minimum value: f(1) = 0

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(len(x)-1):
                sum += 100*(x[i+1] - (x[i]**2))**2 + (x[i] - 1)**2
            return sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0,
                         name="Generalised Rosenbrock's Function")

    @staticmethod
    def f6(domain=(-100, 100), dimensions=30):
        """
        Step Function
        Domain: -100 -> 100
        Minimum value: f(p) = 0, -0.5 <= p <= 0.5

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(len(x)):
                sum += (abs(x[i] + 0.5))**2
            return sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0, name="Step Function")

    @staticmethod
    def f7(domain=(-1.28, 1.28), dimensions=30):
        """
        Quartic Function i.e. Noise
        Domain: -1.28 -> 1.28
        Minimum value: f(0) = 0

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(len(x)):
                sum += (i + 1) * x[i]**4
            sum += random.uniform(0, 1)
            return sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0,
                         name="Quartic Function i.e. Noise")

    @staticmethod
    def f8(domain=(-500, 500), dimensions=30):
        """
        Generalised Schwefel's Problem 2.26
        Domain: -500 -> 500
        Minimum value: f(420.97) = 12569.5(30)/41898.3(100)

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(len(x)):
                sum += -x[i] * math.sin(math.sqrt(abs(x[i])))
            return sum
        min_value = -(418.983 * dimensions)
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=min_value,
                         name="Generalised Schwefel's Problem 2.26")

    @staticmethod
    def f9(domain=(-5.12, 5.12), dimensions=30):
        """
        Generalised Rastrigin's Function
        Domain: -5.12 -> 5.12
        Minimum value: f(0) = 0

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(len(x)):
                sum += (x[i]**2) - 10*math.cos(2*math.pi*x[i]) + 10
            return sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0,
                         name="Generalised Rastrigin's Function")

    @staticmethod
    def f10(domain=(-32, 32), dimensions=30):
        """
        Ackley's Function
        Domain: -32 -> 32
        Minimum value: f(0) = 0

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            n = len(x)
            for i in range(n):
                a = 0
                b = 0
                for i in range(n):
                    a += x[i]**2
                    b += math.cos(2*math.pi*x[i])
                sum += -20*math.exp(-0.2*math.sqrt((1/n)*a)) - math.exp((1/n)*b) + 20 + math.e
            return sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0, name="Ackley's Function")

    @staticmethod
    def f11(domain=(-600, 600), dimensions=30):
        """
        Generalised Griewank Function
        Domain: -600 -> 600
        Minimum value: f(0) = 0

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            prod = 0
            for i in range(len(x)):
                sum += x[i]**2
                prod *= math.cos(x[i]/(math.sqrt(i+1)))
            total = (1/4000)*sum + prod
            return total
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0,
                         name="Generalised Griewank Function")

    @staticmethod
    def f12(domain=(-50, 50), dimensions=30):
        """
        Generalised Penalised Function
        Domain: -50 -> 50
        Minimum value: f(-1) = 0

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            n = len(x)
            sum = 0
            secondsum = 0
            for i in range(n):
                if i < n - 1:
                    sum += ((((ComparativeBenchmarks.y(x[i]) - 1)**2) * (1 + 10*(math.sin(math.pi *
                            ComparativeBenchmarks.y(x[i+1])))**2)) + (ComparativeBenchmarks.y(x[n-1])-1)**2)
                secondsum += ComparativeBenchmarks.u(x[i], 10, 100, 4)
            total = (math.pi/n)*((10*math.sin(math.pi*ComparativeBenchmarks.y(x[1])))**2 + sum) + secondsum
            return total
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=0,
                         name="Generalised Penalised Function")

    @staticmethod
    def f13(domain=(-50, 50), dimensions=30):
        """
        TODO CHECK AGAIN
        Generalised Penalised Function
        Domain: -50 -> 50
        Minimum value: f(1,...,1,-4.76) = -1.1428

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            n = len(x)
            sum = 0
            secondsum = 0
            for i in range(n):
                if i < n - 1:
                    sum += (((x[i] - 1)**2) * (1 + (math.sin(3*math.pi*x[i+1]))**2)) + (x[n-1] - 1) * \
                           (1 + (math.sin(2*math.pi*x[n-1]))**2)
                secondsum += ComparativeBenchmarks.u(x[i], 5, 100, 4)
            total = 0.1*((math.sin(3*math.pi*x[1]))**2 + sum) + secondsum
            return total
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=-1.1428,
                         name="Generalised Penalised Function")

    @staticmethod
    def f14(domain=(-65.54, 65.54), dimensions=2):
        """
        M. Shekel's Foxholes Function
        Dimensions: 2
        Domain: -65.54 -> 65.54
        Minimum value: f(31.95) = 0.998 (~1)

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for j in range(24):
                secondsum = 0
                for i in range(1):
                    secondsum += (x[i] + ComparativeBenchmarks.a_matrix[i][j])**6
                sum += (j + 1 + secondsum)**-1
            total = ((1/500) + sum)**-1
            return total
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=-1.1428,
                         name="M. Shekel's Foxholes Function")

    @staticmethod
    def f15(domain=(-5, 5), dimensions=4):
        """
        N. Kowalik's Function
        Dimensions: 4
        Domain: -5 -> 5
        Minimum value: f(0.19,0.19,0.12,0.14) = 0.0003075

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(10):
                sum += (ComparativeBenchmarks.a_vector[i] -
                        ((x[0]*((ComparativeBenchmarks.b_vector[i]**2)+ComparativeBenchmarks.b_vector[i]*x[1])) /
                         ((ComparativeBenchmarks.b_vector[i]**2) +
                          ComparativeBenchmarks.b_vector[i]*x[2]+x[3])))**2
            return sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=-1.1428,
                         name="N. Kowalik's Function")

    @staticmethod
    def f16(domain=(-5, 5), dimensions=2):
        """
        O. Six-Hump Camel-Back Function
        Dimensions: 2
        Domain: -5 -> 5
        Minimum value: f(-0.09,0.71) = -1.0316

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            return 4*x[0]**2 - 2.1*x[0]**4 + (1/3)*x[0]**6 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=-1.0316,
                         name="O. Six-Hump Camel-Back Function")

    @staticmethod
    def f17(domain=(-5, 15), dimensions=2):
        """
        P. Branin Function
        Dimensions: 2
        Domain: -5 -> 15
        Minimum value: f(9.42,2.47) = 0.398

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            return (x[1] - (5.1/4*(math.pi**2))*(x[0]**2) + (5/math.pi)*x[0] - 6)**2 + 10*(1 - (1/(8*math.pi))) * \
                   math.cos(x[0])+10
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=-1.0316,
                         name="P. Branin Function")

    @staticmethod
    def f18(domain=(-2, 2), dimensions=2):
        """
        Q. Goldstein-Price Function
        Dimensions: 2
        Domain: -2 -> 2
        Minimum value: f(0,-1) = 3

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            return ((1+((x[0]+x[1]+1)**2)*(19-(14*x[0])+((3*x[0])**2)-(14*x[1])+(6*x[0]*x[1])+((3*x[1])**2))) *
                    (30+(((2*x[0])-(3*x[1]))**2)*(18-(32*x[0])+(12*(x[0]**2))+(48*x[1])-(36*x[0]*x[1])+(27*(x[1]**2)))))
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=3,
                         name="Q. Goldstein-Price Function")

    @staticmethod
    def f19(domain=(0, 1), dimensions=3):
        """
        R. Hartman's Family
        Dimensions: 3
        Domain: 0 -> 1
        Minimum value: f(0.114,0.556,0.852) = -3.86

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(4):
                secondsum = 0
                for j in range(3):
                    secondsum += ComparativeBenchmarks.a_hartman19[i][j] * \
                                 ((x[j]-ComparativeBenchmarks.p_hartman19[i][j])**2)
                sum += ComparativeBenchmarks.c_hartman[i]*math.exp(-secondsum)
            return -sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=-3.86,
                         name="R. Hartman's Family")

    @staticmethod
    def f20(domain=(0, 1), dimensions=6):
        """
        R. Hartman's Family
        Dimensions: 6
        Domain: 0 -> 1
        Minimum value: f(0.201,0.150,0.477,0.275,0.311,0.657) = -3.32

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(4):
                secondsum = 0
                for j in range(3):
                    secondsum += ComparativeBenchmarks.a_hartman20[i][j] * \
                                 ((x[j]-ComparativeBenchmarks.p_hartman20[i][j])**2)
                sum += ComparativeBenchmarks.c_hartman[i]*math.exp(-secondsum)
            return -sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=-3.32,
                         name="R. Hartman's Family")

    @staticmethod
    def f21(domain=(0, 10), dimensions=4):
        """
        S. Shekel's Family
        Dimensions: 4
        Domain: 0 -> 10
        Minimum value: f(~4) = -10.2

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(4):
                vector_1 = ComparativeBenchmarks.subtract_vectors(x, ComparativeBenchmarks.a_shekel[i])
                vector_2 = ComparativeBenchmarks.subtract_vectors(x, ComparativeBenchmarks.a_shekel[i])
                sum += (ComparativeBenchmarks.dot_product(vector_1, vector_2) + ComparativeBenchmarks.c_shekel[i])**-1
            return -sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=-10.2,
                         name="S. Shekel's Family")

    @staticmethod
    def f22(domain=(0, 10), dimensions=4):
        """
        S. Shekel's Family
        Dimensions: 4
        Domain: 0 -> 10
        Minimum value: f(~4) = -10.4

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(6):
                vector_1 = ComparativeBenchmarks.subtract_vectors(x, ComparativeBenchmarks.a_shekel[i])
                vector_2 = ComparativeBenchmarks.subtract_vectors(x, ComparativeBenchmarks.a_shekel[i])
                sum += (ComparativeBenchmarks.dot_product(vector_1, vector_2) + ComparativeBenchmarks.c_shekel[i]) ** -1
            return -sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=-10.4,
                         name="S. Shekel's Family")

    @staticmethod
    def f23(domain=(0, 10), dimensions=4):
        """
        S. Shekel's Family
        Dimensions: 4
        Domain: 0 -> 10
        Minimum value: f(~4) = -10.5

        :param domain:
        :param dimensions:
        :return:
        """
        def function(x):
            sum = 0
            for i in range(9):
                vector_1 = ComparativeBenchmarks.subtract_vectors(x, ComparativeBenchmarks.a_shekel[i])
                vector_2 = ComparativeBenchmarks.subtract_vectors(x, ComparativeBenchmarks.a_shekel[i])
                sum += (ComparativeBenchmarks.dot_product(vector_1, vector_2) + ComparativeBenchmarks.c_shekel[i]) ** -1
            return -sum
        return Benchmark(function=function, domain=domain, dimensions=dimensions, min_value=-10.5,
                         name="S. Shekel's Family")


if __name__ == "__main__":
    for function in ComparativeBenchmarks.benchmarks():
        print(function.name)
