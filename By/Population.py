from By.Individual import Individual
import copy
from abc import ABC


class Population(ABC):

    def __init__(self, size, eval_limit, benchmark):
        """
        Represents base class for evolutionary algorithms.
        Creates population of given size with solutions randomly initialised.
        Solutions are evaluated against benchmark function.


        :param size:
        :param eval_limit:
        :param benchmark:
        """
        self._individuals = [None] * size
        self._best_individual = None
        self._size = size
        self._dimensions = benchmark.dimensions
        self._domain = benchmark.domain
        self._function = benchmark.function
        self._min_value = benchmark.min_value
        self._eval_limit = eval_limit
        self._evaluations = 0
        self.random_population()

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
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain

    @property
    def function(self):
        return self._function

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, min_value):
        self._min_value = min_value

    @property
    def eval_limit(self):
        return self._eval_limit

    @eval_limit.setter
    def eval_limit(self, eval_limit):
        self._eval_limit = eval_limit

    @property
    def evaluations(self):
        return self._evaluations

    @evaluations.setter
    def evaluations(self, evaluations):
        self._evaluations = evaluations

    def set_populations_fitness(self):
        """
        Sets all individuals fitness, auto updating best solution and best fitness for individual and population.
        """
        for i in range(self.size):
            self.set_fitness(i)

    def set_fitness(self, ind, value=None, fitness=None):
        """
        Sets individuals function value and fitness attributes.
        Automatically updates best solution and best fitness for individual and population.

        :param ind:     index of individual in poulation list, or individual object
        :param value:   solutions function value, sets automatically if not known
        :param fitness: solutions fitness value, sets automatically if not known
        """
        if isinstance(ind, int):
            ind = self.individuals[ind]
        if value is None:
            value = self.get_solutions_value(ind.solution)
        ind.value = value
        if fitness is None:
            fitness = self.get_fitness(ind.value)
        ind.fitness = fitness
        if ind.best_fitness is None or ind.best_fitness < ind.fitness:
            ind.best_solution = copy.deepcopy(ind.solution)
            ind.best_fitness = copy.deepcopy(ind.fitness)
        if self.best_individual is None or ind.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(ind)

    def get_solutions_value(self, solution):
        """
        Get solutions function value.

        :param solution: solution vector to objective function
        :return: solutions function value
        """
        self.evaluations += 1
        return self.function(solution)

    def random_solution(self, ind):
        """
        Randomises individuals solution vector.

        :param ind: individuals index in population list, or individual object
        """
        if isinstance(ind, int):
            ind = self.individuals[ind]
        ind.random_solution()

    @staticmethod
    def get_fitness(value):
        """
        Get fitness of function value. Fitness values are relative to objective function.

        :param value: float, a solutions function value
        :return: float, a solutions fitness
        """
        if value >= 0:
            fitness = 1 / (1 + value)
        else:
            fitness = 1 + abs(value)
        return fitness

    def get_solutions_fitness(self, solution):
        """
        Get fitness of solution regarding objective function.

        :param solution: solution vector
        :return: float, solutions fitness
        """
        value = self.get_solutions_value(solution)
        return self.get_fitness(value)

    def random_population(self):
        """
        Randomly initialises individuals in the search-space and evaluates their fitnesses.
        """
        for i in range(self.size):
            self.individuals[i] = self.create_individual()
            self.set_fitness(i)

    def create_individual(self):
        """
        Creates a individual objects with a randomly initialised solution and velocity vector.
        :return: individual object
        """
        return Individual(self.dimensions, self.domain)

    def sort_by_fitness(self, increasing_order=True):
        """
        Sorts individuals by fitness in increasing order, if True.

        :param increasing_order: boolean
        :return: sorted list of individuals
        """
        sorted_individuals = self.individuals.copy()
        if increasing_order:
            sorted_individuals.sort(key=lambda i: i.fitness, reverse=False)
        else:
            sorted_individuals.sort(key=lambda i: i.fitness, reverse=True)
        self.individuals = sorted_individuals

    def solution_precision(self, precision):
        """
        Check if required solution precision has been met.

        :param precision:
        :return: boolean
        """
        precision_met = False
        if self.min_value < 0 and self.best_individual.value < self.min_value - (1 * 10 ** precision):
            precision_met = True
        elif self.best_individual.value < self.min_value + (1 * 10 ** -precision):
            precision_met = True
        return precision_met

    def bind(self, value):
        """
        Keeps value within the domain.

        :param value:   float
        :return:    float
        """
        if value < self.domain[0]:
            value = self.domain[0]
        elif value > self.domain[1]:
            value = self.domain[1]
        return value

    def __str__(self):
        str = "[->"
        for ind in self.individuals:
            str += ind.__str__()
            str += ", ->"
        str += "]"
        return str
