from By.Individual import Individual
import numpy.random as random
import copy
from abc import ABC


class Population(ABC):

    def __init__(self, size, dimensions, precision, domain, function):
        self._individuals = [None] * size
        self._best_individual = None
        self._size = size
        self._dimensions = dimensions
        self._precision = precision
        self._domain = domain
        self._function = function
        self._evaluations = 0
        self._random_population()

    @property
    def individuals(self):
        return self._individuals

    @individuals.setter
    def individuals(self, individuals):
        self._individuals = individuals

    @property
    def best_individual(self):
        return self._best_individual

    @best_individual.setter
    def best_individual(self, best_individual):
        self._best_individual = best_individual

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions):
        self._dimensions = dimensions

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        self._precision = precision

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain

    @property
    def function(self):
        return self._function

    @property
    def evaluations(self):
        return self._evaluations

    @evaluations.setter
    def evaluations(self, evaluations):
        self._evaluations = evaluations

    def reset(self):
        self.__init__(self.size, self.dimensions, self.precision, self.domain, self.function)

    def set_populations_fitness(self):
        for i in range(self.size):
            self.set_fitness(i)

    def set_fitness(self, ind):
        if isinstance(ind, int):
            ind = self.individuals[ind]
        ind.value = self.get_solutions_value(ind.solution)
        ind.fitness = self.get_fitness(ind.value)
        self.evaluations += 1
        if self.best_individual is None or ind.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(ind)

    def get_solutions_value(self, solution):
        return self.function(solution)

    def random_solution(self, index):
        ind = self.individuals[index]
        for d in range(self.dimensions):
            ind.solution[d] = random.uniform(self.domain[0], self.domain[1])

    @staticmethod
    def get_fitness(value):
        if value >= 0:
            fitness = 1 / (1 + value)
        else:
            fitness = 1 + abs(value)
        return fitness

    def get_solutions_fitness(self, solution):
        value = self.get_solutions_value(solution)
        return self.get_fitness(value)

    def _random_population(self):
        for i in range(self.size):
            self.individuals[i] = self.create_individual()
            self.set_fitness(i)

    def create_individual(self):
        return Individual(self.dimensions, self.domain)

    def sort_by_fitness(self, increasing_order=True):
        sorted_individuals = self.individuals.copy()
        if increasing_order:
            sorted_individuals.sort(key=lambda i: i.fitness, reverse=False)
        else:
            sorted_individuals.sort(key=lambda i: i.fitness, reverse=True)
        self.individuals = sorted_individuals

    def solution_precision(self, min_value):
        precision_met = False
        if min_value < 0 and self.best_individual.value < min_value - (1 * 10 ** self.precision):
            precision_met = True
        elif self.best_individual.value < min_value + (1 * 10 ** -self.precision):
            precision_met = True
        return precision_met

    def bind(self, x):
        """
        Keeps x within the domain.

        :param x:   float
        :return:    float
        """
        if x < self.domain[0]:
            return self.domain[0]
        elif x > self.domain[1]:
            return self.domain[1]
        else:
            return x

    def __str__(self):
        str = "[->"
        for ind in self.individuals:
            str += ind.__str__()
            str += ", ->"
        str += "]"
        return str
