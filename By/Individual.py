import numpy.random as random


class Individual(object):

    def __init__(self, dimensions, domain):
        self._solution = [None] * dimensions
        self._velocity = [None] * dimensions
        self._random_solution(dimensions, domain)
        self._random_velocity(dimensions, domain)
        self._best_solution = self.solution
        self._fitness = None
        self._best_fitness = None
        self._value = None
        self._counter = 0

    @property
    def solution(self):
        return self._solution

    @solution.setter
    def solution(self, solution):
        self._solution = solution

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    @property
    def best_solution(self):
        return self._best_solution

    @best_solution.setter
    def best_solution(self, best_solution):
        self._best_solution = best_solution

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, new_fitness):
        self._fitness = new_fitness

    @property
    def best_fitness(self):
        return self._best_fitness

    @best_fitness.setter
    def best_fitness(self, best_fitness):
        self._best_fitness = best_fitness

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def counter(self):
        return self._counter

    @counter.setter
    def counter(self, counter):
        self._counter = counter

    def _random_solution(self, dimensions, domain):
        for d in range(dimensions):
            self.solution[d] = random.uniform(domain[0], domain[1])

    def _random_velocity(self, dimensions, domain):
        for d in range(dimensions):
            self.velocity[d] = random.uniform(-abs(domain[1] - domain[0]), abs(domain[1] - domain[0]))

    def __eq__(self, other):
        n = len(self.solution)
        if n != len(other.solution):
            return False
        if self.solution != other.solution:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        str = "( Solution: ["
        for gene in self.solution:
            str += repr(gene)
            str += "; "
        str += "], Fitness: "
        str += repr(self.fitness)
        str += ", Value: "
        str += repr(self.value)
        str += ")"
        return str