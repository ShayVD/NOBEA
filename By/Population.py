from By.Individual import Individual


class Population(object):

    def __init__(self, size, dimensions, precision, domain, function):
        self._individuals = [None] * size
        self._best_individual = None
        self._size = size
        self._dimensions = dimensions
        self._precision = precision
        self._domain = domain
        self._function = function
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

    def reset(self):
        self.__init__(self.size, self.dimensions, self.precision, self.domain, self.function)

    def set_populations_fitness(self):
        for individual in self.individuals:
            self.set_individuals_fitness(individual)

    def set_individuals_fitness(self, individual):
        individual.value = self.get_solutions_value(individual.solution)
        individual.fitness = self.get_values_fitness(individual.value)
        if self.best_individual is None or individual.fitness > self.best_individual.fitness:
            self.best_individual = individual

    def get_solutions_value(self, solution):
        return self.function(solution)

    @staticmethod
    def get_values_fitness(value):
        if value >= 0:
            fitness = 1 / (1 + value)
        else:
            fitness = 1 + abs(value)
        return fitness

    def get_solutions_fitness(self, solution):
        value = self.get_solutions_value(solution)
        return self.get_values_fitness(value)

    def _random_population(self):
        for i in range(self.size):
            individual = self.create_individual()
            self.set_individuals_fitness(individual)
            self.individuals[i] = individual

    def create_individual(self):
        return Individual(self.dimensions, self.domain)

    def sort_by_fitness(self):
        sorted_individuals = self.individuals.copy()
        sorted_individuals.sort(key=lambda i: i.fitness, reverse=True)
        self.individuals = sorted_individuals

    def __str__(self):
        str = "[->"
        for ind in self.individuals:
            str += ind.__str__()
            str += ", ->"
        str += "]"
        return str
