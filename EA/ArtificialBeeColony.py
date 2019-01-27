from NO.ComparativeBenchmarks import ComparativeBenchmarks
from By.Population import Population
import numpy.random as random
import copy


class ArtificialBeeColony(object):

    def __init__(self, generations=100, limit=20, population=None):
        self._generations = generations
        self._limit = limit
        self._population = population
        if self.population is not None:
            self.population.size = int(self.population.size + self.population.size % 2)

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, generations):
        self._generations = generations

    @property
    def limit(self):
        return self._limit

    @limit.setter
    def limit(self, limit):
        self._limit = limit

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        """
        Set population object of the ABC.
        If the population size is uneven then the size is increased by one to fix.

        :param population:  Population object
        """
        if population is not None:
            population.size = int(population.size + population.size % 2)
        self._population = population

    def colonise(self, min_value=None, print_steps=True):
        # Employed bees are half the colony size
        employed = int(self.population.size / 2)
        for generation in range(self.generations):
            # Employees phase
            for e in range(self.population.size):
                self.send_employee(e)
            # Onlookers phase
            self.send_onlookers()
            # Scouts phase
            self.send_scout()
            if print_steps:
                print("Generation:", generation+1, "/", self.generations, ";Solution:", self.population.best_individual)
            if min_value is not None:
                if min_value < 0 and self.population.best_individual.value < min_value-(1*10**-self.population.precision):
                    break
                elif self.population.best_individual.value < min_value+(1*10**-self.population.precision):
                    break
        return self.population.best_individual

    def send_scout(self):
        counters = []
        for bee in self.population.individuals:
            counters += [bee.counter]
        index = counters.index(max(counters))
        if counters[index] > self.limit:
            bee = self.population.create_individual()
            self.population.set_individuals_fitness(bee)
            self.population.individuals[index] = bee
            self.send_employee(index)

    def send_onlookers(self):
        probabilities = self.probabilities()
        beta = 0
        for o in range(self.population.size):
            phi = random.random_sample()
            beta += phi * max(probabilities)
            beta %= max(probabilities)
            index = self.select(beta)
            self.send_employee(index)

    def select(self, beta):
        probabilities = self.probabilities()
        for i in range(self.population.size):
            if beta < probabilities[i]:
                return i

    def send_employee(self, index):
        bee = self.population.individuals[index]
        food_source = self.new_food_source(index)
        nectar = self.population.get_solutions_fitness(food_source)
        if bee.fitness < nectar:
            bee.solution = food_source
            bee.counter = 0
            self.population.set_individuals_fitness(bee)
        else:
            bee.counter += 1

    def probabilities(self):
        # Probabilities computed as Karaboga did in his implementation
        fitnesses = []
        for bee in self.population.individuals:
            fitnesses += [bee.fitness]
        max_fitness = max(fitnesses)
        probs = []
        for f in fitnesses:
            probs += [0.9 * f / max_fitness + 0.1]
        return probs

    def new_food_source(self, bee_index):
        d = random.randint(0, self.population.dimensions)
        phi = random.uniform(-1, 1)
        other_bee_index = bee_index
        while bee_index == other_bee_index:
            other_bee_index = random.randint(0, self.population.size)
        bee = self.population.individuals[bee_index]
        other_bee = self.population.individuals[other_bee_index]
        food_source = copy.deepcopy(bee.solution)
        food_source[d] = bee.solution[d] + phi * (bee.solution[d] - other_bee.solution[d])
        return food_source

    """TODO check if can use Vimal Nayak implementation
        def probabilities(self):
            probabilities = []
            fitnesses = []
            fitness_sum = 0
            for bee in self.population.individuals:
                fitnesses += [bee.fitness]
                fitness_sum += bee.fitness
            for fitness in fitnesses:
                probabilities += [fitness/fitness_sum]
            return probabilities"""


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f1()
    population = Population(size=100, dimensions=3, precision=6, domain=benchmark.domain, function=benchmark.function)
    abc = ArtificialBeeColony(generations=100, limit=20, population=population)

    bee = abc.colonise(min_value=benchmark.min_value, print_steps=True)
    print("Best Solution: ", bee)
