from evolutionary_algorithms.Individual import *
import numerical_optimisation_benchmarks.numb as nb

class Population(object):

    def __init__(self, size, genes, precision, domain):
        self._agents = []
        self._best_global_solution = None
        self._size = size
        self._genes = genes
        self._precision = precision
        self._domain = domain
        self._fitness = nb.multi_michalewicz
        self.initial_population()

    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, agents):
        self._agents = agents

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, genes):
        self._genes = genes

    @property
    def best_global_solution(self):
        return self._best_global_solution

    @best_global_solution.setter
    def best_global_solution(self, best_solution):
        self._best_global_solution = best_solution

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

    def initial_population(self):
        for i in range(self.size):
            ind = Individual()
            ind.initial_chromosome(self.genes, self.precision, self.domain)
            ind.fitness = self.get_fitness(ind)
            self.agents += [ind]

    def set_fitness(self):
        for ind in self.agents:
            ind.fitness = self.get_fitness(ind)

    def get_fitness(self, individual):
        return round(self._fitness(individual.chromosome), self.precision)

    def set_best_positions(self):
        for ind in self.agents:
            ind.best_position = ind.chromosome

    def set_best_global_solution(self):
        for ind in self.agents:
            if self._best_global_solution is None or \
                    ind.fitness < self._best_global_solution.fitness:
                self._best_global_solution = ind

    def set_initial_velocity(self):
        for ind in self.agents:
            ind.initial_velocity(self.genes, self.precision, self.domain)

    def set_initial_fitness(self):
        for ind in self.agents:
            ind.fitness = self.get_fitness(ind)
            ind.best_fitness = ind.fitness

    def sort_by_fitness(self):
        sorted = self.agents.copy()
        sorted.sort(key=lambda i: i.fitness)
        self.agents = sorted

    def best_fitness(self):
        best = None
        for ind in self.agents:
            if best is None or ind.fitness < best.fitness:
                best = ind
        return best

    def __str__(self):
        str = "[->"
        for ind in self.agents:
            str += ind.__str__()
            str += ", ->"
        str += "]"
        return str
