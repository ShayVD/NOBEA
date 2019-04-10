from NO.ComparativeBenchmarks import ComparativeBenchmarks
from By.Population import Population
import numpy.random as random
import copy


class ArtificialBeeColony(Population):

    def __init__(self, employees, limit, size, eval_limit, benchmark):
        """
        Uses bees food foraging techniques to approximate global optimum of benchmark function.

        :param employees:
        :param limit:
        :param size:
        :param eval_limit:
        :param benchmark:
        """
        super().__init__(size=size, eval_limit=eval_limit, benchmark=benchmark)
        self._limit = limit
        self._employees = round(employees * self.size)
        self._onlookers = self.size - self._employees

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

    def colonise(self, precision=None, print_steps=True):
        """
        Colonise the search-space.

        :param precision:
        :param print_steps:
        :return:
        """
        generation = 0
        self._mins = []
        while self.evaluations < self.eval_limit:
            # Employees phase
            for e in range(self.employees):
                self.send_employee(e)
            # Onlookers phase
            self.send_onlookers()
            # Scouts phase
            self.send_scouts()
            if print_steps:
                print("Generation:", generation+1, "; Evaluations: ", self.evaluations,
                      "; Solution:", self.best_individual)
            if precision is not None and self.solution_precision(precision):
                break
            generation += 1
        return self.best_individual, self._mins

    def send_employee(self, index, other_index=None):
        """
        Exploit a food source area. If only one index is given, then that bee explores its own food source.
        If two are given, then the first bee explores the other bees food source.
        If the source has more nectar, then stay, and reset counter.
        Otherwise return to previous source, and increase counter by one.

        :param index:   int, index of the bee in the population list
        :param other_index: int, index of the other bee in the population list
        """
        bee = self.individuals[index]
        if other_index is not None:
            food_source = self.explore_food_source(other_index)
        else:
            food_source = self.explore_food_source(index)
        value = self.get_solutions_value(food_source)
        nectar = self.get_fitness(value)
        if self.evaluations % 100 == 0:
            self._mins += [self.best_individual.value]
        if bee.fitness < nectar:
            bee.solution = food_source
            self.reset_counter(index)
            self.set_fitness(index, value, nectar)
        else:
            bee.counter += 1

    def send_onlookers(self):
        """
        Send onlooker bees to exploit the best food sources.
        """
        for o in range(self.employees, self.size):
            best_food_sources = self.best_food_sources()
            employee_index = random.choice(best_food_sources)
            self.send_employee(o, employee_index)

    def best_food_sources(self):
        """
        Select the best food sources based on the employees waggle dance.

        :return: list of employees index
        """
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
        Send the bee with the highest count to a new food source if they have gone over the limit.
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
        """
        Reset a bees counter.

        :param index: bees index
        """
        self.individuals[index].counter = 0

    def probabilities(self):
        """
        Produce the probabilities that an employees food source will be picked to exploit by an onlooker.
        Relative to the max fitness.

        :return: list of probabilities
        """
        probs = []
        max_fitness = max(self.individuals, key=lambda bee: bee.fitness)
        for bee in self.individuals:
            probs += [0.9 * bee.fitness / max_fitness.fitness + 0.1]
        return probs

    def explore_food_source(self, bee_index):
        """
        Produce a food source in the area of the given bees food source.

        :param bee_index: int, bees index
        :return: food source
        """
        d = random.randint(0, self.dimensions)
        rand = random.uniform(-1, 1)
        other_bee_index = bee_index
        while bee_index == other_bee_index:
            fs = self.best_food_sources()
            other_bee_index = random.choice(fs)
            #other_bee_index = random.randint(0, self.size)
        solution = self.individuals[bee_index].solution
        other_solution = self.individuals[other_bee_index].solution
        food_source = copy.deepcopy(solution)
        food_source[d] = self.bind(solution[d] + rand * (solution[d] - other_solution[d]))
        return food_source


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f1()
    # benchmark.domain = [-10, 10]
    # benchmark.dimensions = 10
    abc = ArtificialBeeColony(employees=0.5, limit=100, size=100, eval_limit=100000, benchmark=benchmark)
    bee, mins = abc.colonise(precision=2, print_steps=False)
    print("Best Solution: ", bee)
