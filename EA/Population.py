from EA.Individual import *
from math import pi
import NO.numb as nb


class Population(object):

    def __init__(self, size=100, genes=3, precision=6, domain=[0, pi], fitness_function=nb.multi_michalewicz):
        """
        Creates a population of 'size' individuals, with 'genes' genes up to a precision of 'precision',
        in the domain 'domain[0]' to 'domain[1]'.
        Fitness of each individual is calculated using 'fitness_function'.
        Individual with lowest fitness score (min) becomes best individual (global).

        :param size:            int, amount of individuals in the population
        :param genes:           int, amount of genes of each individual
        :param precision:       int, amount of digits after the decimal point the genes must be accurate to
        :param domain:          [number, number], list containing lower and higher boundary of domain
        :param fitness_function:function, used to calculate fitness of individuals
        """
        self._individuals = [None] * size
        self._best_individual = None
        self._size = size
        self._genes = genes
        self._precision = precision
        self._domain = domain
        self._fitness_function = fitness_function
        self.initial_population()

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
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, genes):
        self._genes = genes

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
        """
        Call during '__init__'.
        Create Individuals.
        Set fitness' of Individuals.
        Update best individual of population, if needed.
        """
        for i in range(self.size):
            ind = Individual(self.genes, self.precision, self.domain)
            ind.set_fitness(self.get_fitness(ind))
            if self.best_individual is None or ind.fitness < self.best_individual.fitness:
                self.best_individual = ind
            self.individuals[i] = ind

    def set_population_fitness(self):
        """
        Get and then set fitness of every individual in the population.
        Update best individual of population, if needed.
        """
        for ind in self.individuals:
            fitness = self.get_fitness(ind)
            ind.set_fitness(fitness)
            if ind.fitness < self.best_individual.fitness:
                self.best_individual = ind

    def get_fitness(self, individual):
        """
        Get fitness of 'individual' using the fitness function.

        :param individual: Individual object
        :return: float, fitness score
        """
        return round(self._fitness_function(individual.chromosome), self.precision)

    def sort_by_fitness(self):
        """
        Sort individuals in population by fitness, from lowest to highest.
        """
        sorted_individuals = self.individuals.copy()
        sorted_individuals.sort(key=lambda i: i.fitness)
        self.individuals = sorted_individuals

    def __str__(self):
        str = "[->"
        for ind in self.individuals:
            str += ind.__str__()
            str += ", ->"
        str += "]"
        return str
