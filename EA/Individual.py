import random
class Individual(object):

    def __init__(self):
        self._chromosome = []
        self._fitness = None
        self._best_position = None
        self._best_fitness = None
        self._velocity = []

    @property
    def chromosome(self):
        return self._chromosome

    @chromosome.setter
    def chromosome(self, new_chromosome):
        self._chromosome = new_chromosome

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, new_fitness):
        self._fitness = new_fitness

    @property
    def best_position(self):
        return self._best_position

    @best_position.setter
    def best_position(self, best_position):
        self._best_position = best_position

    @property
    def best_fitness(self):
        return self._best_fitness

    @best_fitness.setter
    def best_fitness(self, best_fitness):
        self._best_fitness = best_fitness

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    def initial_chromosome(self, genes, precision, domain):
        self.chromosome = []
        for i in range(genes):
            self.chromosome += [round(random.uniform(domain[0], domain[1]), precision)]
        self.best_position = self.chromosome

    def initial_velocity(self, genes, precision, domain):
        self.velocity = []
        for i in range(genes):
            self.velocity += [round(random.uniform(-abs(domain[1] - domain[0]), abs(domain[1] - domain[0])), precision)]

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
        str = "(["
        for gene in self.chromosome:
            str += repr(gene)
            str += "; "
        str += "], "
        str += repr(self.fitness)
        str += ")"
        return str