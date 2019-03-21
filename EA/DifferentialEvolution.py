from By.Population import Population
from NO.ComparativeBenchmarks import ComparativeBenchmarks
import numpy as np


class DifferentialEvolution(Population):
    """
    DE is a meta-heuristic that optimises a problem by iteratively trying to improve candidate solutions, agents,
    with regard to a given measure of quality, the fitness function.
    A population of agents is maintained.
    New agents are created each generation by combining existing ones according to it's simple formulae,
    and then keeps whichever agent has the best score, or fitness, on the optimisation problem.
    In each iteration the fitness of each agent is calculated using the fitness function.
    When a satisfactory solution is found to the fitness function, or max iterations is reached, the algorithm stops.
    """

    def __init__(self, crossover, mutate, size, generations, dimensions, domain, precision, function):
        """
        Set class attributes.

        :param generations:     int, amount of iterations the algorithm will run
        :param crossover:       float, probability of crossover
        :param mutate:          float, probability of mutation
        """
        super().__init__(size=size, dimensions=dimensions, precision=precision, domain=domain, function=function)
        self._generations = generations
        self._crossover = crossover
        self._mutate = mutate

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, generations):
        self._generations = generations

    @property
    def crossover(self):
        return self._crossover

    @crossover.setter
    def crossover(self, crossover):
        self._crossover = crossover

    @property
    def mutate(self):
        return self._mutate

    @mutate.setter
    def mutate(self, mutate):
        self._mutate = mutate

    def evolve(self, min_value=None, print_steps=True):
        """
        Evolve the population.

        :param print_steps:     boolean, prints best solution at each step
        :return:                individual, best individual after max iterations or required fitness
        """
        for generation in range(self.generations):
            for i in range(self.size):
                self.new_position(i)
            if print_steps:
                print("Generation:", generation+1, "/", self.generations, ";Solution:", self.best_individual)
            if min_value is not None and self.solution_precision(min_value):
                break
        return self.best_individual

    def new_position(self, index):
        """
        Create agent with new position, that will replace given agent if it's fitness is better.

        :param index:   individual, the individual that will be replaced if a better solution is found
        :return:        individual
        """
        agent = self.individuals[index]
        # Pick 3 unique solutions
        solutions = self.unique_agents(index, 3)
        a = self.individuals[solutions[0]].solution
        b = self.individuals[solutions[1]].solution
        c = self.individuals[solutions[2]].solution
        # Pick random index in range of dimensions (genes)
        R = np.random.randint(0, self.dimensions)
        # Compute the new agents position
        solution = [None] * self.dimensions
        for j in range(self.dimensions):
            # For each gene pick a uniformly distributed random number
            ri = round(np.random.random_sample(), 2)
            if ri < self.crossover or j == R:
                solution[j] = self.bind(a[j] + self.mutate * (b[j] - c[j]))
            else:
                solution[j] = agent.solution[j]
        value = self.get_solutions_value(solution)
        fitness = self.get_fitness(value)
        if fitness > agent.fitness:
            agent.solution = solution
            agent.value = value
            agent.fitness = fitness

    def unique_agents(self, index, n):
        """
        Get 3 agents that are unique from each other and 'agent' from the population.

        :param index:
        :return: list of 3 agents
        """
        solutions = []
        while len(solutions) < n:
            i = np.random.randint(self.size)
            if i != index and i not in solutions:
                solutions += [i]
        return solutions


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f1()
    de = DifferentialEvolution( crossover=0.8, mutate=0.2, size=100, generations=100, dimensions=6,
                                domain=benchmark.domain, precision=6, function=benchmark.function)
    agent = de.evolve(min_value=None, print_steps=True)
    print("Best Solution: ", agent)
