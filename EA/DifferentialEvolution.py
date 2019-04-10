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

    def __init__(self, crossover, mutate, size, eval_limit, benchmark):
        """
        Set class attributes.

        :param crossover:
        :param mutate:
        :param size:
        :param eval_limit:
        :param benchmark:
        """
        super().__init__(size=size, eval_limit=eval_limit, benchmark=benchmark)
        self._crossover = crossover
        self._mutate = mutate

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

    def evolve(self, precision=None, print_steps=True):
        """
        Evolve the population.

        :param precision:       precision of required solution
        :param print_steps:     boolean, prints best solution at each step
        :return:                individual, best individual after max iterations or required fitness
        """
        generation = 0
        mins = []
        while self.evaluations < self.eval_limit:
            for i in range(self.size):
                self.new_position(i)
            if print_steps:
                print("Generation:", generation+1, "; Evaluations: ", self.evaluations, "; Solution: ",
                      self.best_individual)
            if precision is not None and self.solution_precision(precision):
                break
            mins += [self.best_individual.value]
            generation += 1
        return self.best_individual, mins

    def new_position(self, index):
        """
        Calculate a new position for the given agent. If the fitness is better then move to the new position.
        Otherwise the agent stays where it is.

        :param index:   agents index
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
            ri = np.random.random_sample()
            if ri < self.crossover or j == R:
                solution[j] = self.bind(a[j] + self.mutate * (b[j] - c[j]))
            else:
                solution[j] = agent.solution[j]
        value = self.get_solutions_value(solution)
        fitness = self.get_fitness(value)
        if fitness > agent.fitness:
            agent.solution = solution
            self.set_fitness(agent, value, fitness)

    def unique_agents(self, index, n):
        """
        Get n agents index from the population that are unique from each other and the given index.

        :param index:   agents index
        :param n:       the number of index required
        :return:        list of index
        """
        solutions = []
        while len(solutions) < n:
            i = np.random.randint(self.size)
            if i != index and i not in solutions:
                solutions += [i]
        return solutions


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f1()
    # benchmark.domain = [-10, 10]
    # benchmark.dimensions = 10
    de = DifferentialEvolution(crossover=0.8, mutate=0.2, size=100, eval_limit=100000, benchmark=benchmark)
    agent, mins = de.evolve(precision=None, print_steps=True)
    print("Best Solution: ", agent)
