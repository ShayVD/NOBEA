import random


class Individual(object):

    def __init__(self, genes, precision, domain):
        """
        Creates individual with 'genes' genes, all up to a precision of 'precision' after the decimal point,
        within the domain 'domain[0]' to 'domain[1]'.
        Space is created for the individuals chromosome and velocity, then both are initialised using the provided info.
        The best chromosome is set to it's initial chromosome.
        The fitness variables will need to be set (by the Population class) as the individual does not hold the fitness
        evaluator.

        :param genes:       int, the amount of genes (dimensions) it has
        :param precision:   int, the precision required for each gene
        :param domain:      [number, number], the lower and higher boundary of where to search for a solution
        """
        self._chromosome = [None] * genes
        self._velocity = [None] * genes
        self._init_chromosome(genes, precision, domain)
        self._init_velocity(genes, precision, domain)
        self._best_chromosome = self.chromosome
        self._fitness = None
        self._best_fitness = None

    @property
    def chromosome(self):
        return self._chromosome

    @chromosome.setter
    def chromosome(self, new_chromosome):
        self._chromosome = new_chromosome

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    @property
    def best_chromosome(self):
        return self._best_chromosome

    @best_chromosome.setter
    def best_chromosome(self, best_chromosome):
        self._best_chromosome = best_chromosome

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

    def _init_chromosome(self, genes, precision, domain):
        """
        Randomly, and uniformly, initialises the chromosome to the given amount of genes, up to the given precision for
        each gene, within the specified domain.

        :param genes:       int, the amount of genes (dimensions) the individual has
        :param precision:   int, the precision required for each gene
        :param domain:      [number, number], the lower and higher boundary of the solution space
        :return:
        """
        for i in range(genes):
            self.chromosome[i] = round(random.uniform(domain[0], domain[1]), precision)

    def _init_velocity(self, genes, precision, domain):
        """
        Randomly, and uniformly, initialises the velocity of each gene to the given precision within the domain.

        :param genes:       int, the amount of genes (dimensions) the individual has
        :param precision:   int, the precision required for each gene
        :param domain:      [number, number], the lower and higher boundary of the solution space
        :return:
        """
        for i in range(genes):
            self.velocity[i] = round(random.uniform(-abs(domain[1] - domain[0]), abs(domain[1] - domain[0])), precision)

    def set_fitness(self, fitness):
        """
        Set fitness of individual with given value. If best fitness variable is not set, or given fitness is better
        (lower) then set best fitness to the new value.

        :param fitness: float, fitness of the individual with regards to some fitness function
        :return:
        """
        self.fitness = fitness
        if self.best_fitness is None or fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_chromosome = self.chromosome

    def __eq__(self, other):
        n = len(self.chromosome)
        if n != len(other.chromosome):
            return False
        if self.chromosome != other.chromosome:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        str = "( Genes: ["
        for gene in self.chromosome:
            str += repr(gene)
            str += "; "
        str += "], Fitness: "
        str += repr(self.fitness)
        str += ")"
        return str