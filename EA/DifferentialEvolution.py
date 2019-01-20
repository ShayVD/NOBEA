from EA.Population import Population
from NO.ComparativeBenchmarks import *
import numpy as np


class DifferentialEvolution(object):
    """
    DE is a meta-heuristic that optimises a problem by iteratively trying to improve candidate solutions, agents,
    with regard to a given measure of quality, the fitness function.
    A population of agents is maintained.
    New agents are created each generation by combining existing ones according to it's simple formulae,
    and then keeps whichever agent has the best score, or fitness, on the optimisation problem.
    In each iteration the fitness of each agent is calculated using the fitness function.
    When a satisfactory solution is found to the fitness function, or max iterations is reached, the algorithm stops.
    """

    def __init__(self, generations=100, crossover=0.9, mutate=0.8, population=None):
        """
        Set class attributes.

        :param generations:     int, amount of iterations the algorithm will run
        :param crossover:       float, probability of crossover
        :param mutate:          float, probability of mutation
        :param population:      Population, population the algorithm will run on
        """
        self._generations = generations
        self._crossover = crossover
        self._mutate = mutate
        self._population = population

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

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        self._population = population

    def evolve(self, min_value=None, print_steps=True):
        """
        Evolve the population.

        :param print_steps:     boolean, prints best solution at each step
        :return:                individual, best individual after max iterations or required fitness
        """
        for generation in range(self.generations):
            for i in range(self.population.size):
                agent = self.population.individuals[i]
                new_agent = self.new_position(agent)
                if new_agent.fitness < agent.fitness:
                    self.population.individuals[i] = new_agent
            if print_steps:
                print("Generation:", generation+1, "/", self.generations, ";Solution:", self.population.best_individual)
            if min_value is not None:
                if min_value < 0 and self.population.best_individual.fitness < min_value-1*10**-self.population.precision:
                    break
                elif self.population.best_individual.fitness < min_value+1*10**-self.population.precision:
                    break
        return self.population.best_individual

    def new_position(self, agent):
        """
        Create agent with new position, that will replace given agent if it's fitness is better.

        :param agent:   individual, the individual that will be replaced if a better solution is found
        :return:        individual
        """
        # Pick 3 unique agents
        agents = self.three_unique_agents(agent)
        a, b, c = agents
        # Pick random index in range of dimensions (genes)
        R = np.random.randint(0, self.population.genes)
        # Compute the new agents position
        new_agent = self.population.create_individual()
        for j in range(self.population.genes):
            # For each gene pick a uniformly distributed random number
            ri = round(np.random.random_sample(), 2)
            if ri < self.crossover or j == R:
                new_agent.chromosome[j] = round(a.chromosome[j] + self.mutate * (b.chromosome[j] - c.chromosome[j]),
                                                self.population.precision)
            else:
                new_agent.chromosome[j] = agent.chromosome[j]
        self.population.set_individual_fitness(new_agent)
        return new_agent

    def three_unique_agents(self, agent):
        """
        Get 3 agents that are unique from each other and 'agent' from the population.

        :param agent:
        :return: list of 3 agents
        """
        agents = []
        while len(agents) < 3:
            i = np.random.randint(0, self.population.size)
            x = self.population.individuals[i]
            if x != agent and x not in agents:
                agents += [x]
        return agents


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f1()
    de = DifferentialEvolution(generations=100, crossover=0.8, mutate=0.6)
    de.population = Population(size=100, genes=3, precision=6, domain=benchmark.domain,
                               fitness_function=benchmark.function)
    agent = de.evolve(min_value=benchmark.min_value, print_steps=True)
    print("Best Solution: ", agent)
