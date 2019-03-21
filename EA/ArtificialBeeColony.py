from NO.ComparativeBenchmarks import ComparativeBenchmarks
from By.Population import Population
import numpy.random as random
import copy


class ArtificialBeeColony(Population):

    def __init__(self, employees, limit, size, generations, dimensions, domain, precision, function):
        """
        Set class attributes

        :param generations:
        :param limit:
        """
        size = int(size+size % 2)
        super().__init__(size=size, dimensions=dimensions, precision=precision, domain=domain, function=function)
        self._generations = generations
        self._limit = limit
        self._employees = round(employees * self.size)
        self._onlookers = self.size - self._employees

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, generations):
        self._generations = generations

    @property
    def employees(self):
        return self._employees

    @employees.setter
    def employees(self, employees):
        self._employees = round(employees * self.size)

    @property
    def onlookers(self):
        return self._onlookers

    @property
    def limit(self):
        return self._limit

    @limit.setter
    def limit(self, limit):
        self._limit = limit

    def colonise(self, min_value=None, print_steps=True):
        """
        Run the ABC algorithm on the given population.

        :param min_value:
        :param print_steps:
        :return:
        """
        for generation in range(self.generations):
            # Employees phase
            for e in range(self.employees):
                self.send_employee(e)
            # Onlookers phase
            self.send_onlookers()
            # Scouts phase
            self.send_scouts()
            if print_steps:
                print("Generation:", generation+1, "/", self.generations, ";Solution:", self.best_individual)
            if min_value is not None and self.solution_precision(min_value):
                break
        return self.best_individual

    def send_employee(self, index, other_index=None):
        """
        Send a bee to a new food source.
        If the source has more nectar then stay, and reset counter.
        Otherwise return to previous source, and increase counter by one.

        :param index:   int, index of the bee in the population list
        """
        bee = self.individuals[index]
        if other_index is not None:
            food_source = self.explore_food_source(other_index)
        else:
            food_source = self.explore_food_source(index)
        nectar = self.get_solutions_fitness(food_source)
        if bee.fitness < nectar:
            bee.solution = food_source
            self.reset_counter(index)
            self.set_fitness(index)
        else:
            bee.counter += 1

    def send_onlookers(self):
        for o in range(self.employees, self.onlookers):
            # get best food sources
            best_food_sources = self.best_food_sources()
            # randomly pick one
            employee_index = random.choice(best_food_sources)
            # send onlooker to food source and evaluate
            self.send_employee(o, employee_index)

    def best_food_sources(self):
        best = []
        while not best:
            probabilities = self.probabilities()
            rand = random.random_sample()
            for e in range(self.employees):
                if probabilities[e] > rand:
                    best += [e]
        return best

    def send_scouts(self):
        """
        Send bees to a new food source if they have gone over the limit.
        """
        max_index = 0
        max_limit = self.individuals[0].counter
        for i in range(1, self.size):
            if self.individuals[i].counter > max_limit:
                max_limit = self.individuals[i].counter
                max_index = i
        if max_limit > self.limit:
            self.random_solution(max_index)
            self.set_fitness(max_index)
            self.reset_counter(max_index)

    def reset_counter(self, index):
        self.individuals[index].counter = 0

    def probabilities(self):
        probs = []
        max_fitness = max(self.individuals, key=lambda bee: bee.fitness)
        for bee in self.individuals:
            probs += [0.9 * bee.fitness / max_fitness + 0.1]
        return probs

    def explore_food_source(self, bee_index):
        d = random.randint(0, self.dimensions)
        rand = random.uniform(-1, 1)
        other_bee_index = bee_index
        while bee_index == other_bee_index:
            other_bee_index = random.randint(0, self.size)
        solution = self.individuals[bee_index].solution
        other_solution = self.individuals[other_bee_index].solution
        food_source = copy.deepcopy(solution)
        food_source[d] = solution[d] + rand * (solution[d] - other_solution[d])
        return food_source


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f1()
    abc = ArtificialBeeColony(employees=0.5, limit=100, size=20, generations=2500, dimensions=30,
                              domain=benchmark.domain, precision=6, function=benchmark.function)
    bee = abc.colonise(min_value=None, print_steps=True)
    print("Best Solution: ", bee)
