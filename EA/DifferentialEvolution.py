from evolutionary_algorithms.Population import *
from math import *

class DifferentialEvolution(object):

    def __init__(self, generations=100, crossover=0.9, mutate=0.8):
        """

        :param precision: int amount of digits after the decimal point
        :param population_size: int amount of individuals in population
        :param domain: list containing low and high boundary of search space
        :param generations: int amount of iterations the algorithm will run
        :param crossover: float probability of crossover
        :param mutate: probability of mutation
        :param genes: amount of genes in an individuals chromosome
        """
        self.generations = generations
        self.crossover = crossover
        self.mutate = mutate
        self._population = None

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        self._population = population

    def new_position(self, i):
        """
        Calculate new position of agent. Return it if fitness is better.
        :param population: list of individuals, current population
        :param i: int, index of individual in population list
        :return: individual
        """
        # Pick 3 unique agents
        agent = self.population.agents[i]
        agents = self.three_unique_agents(agent)
        a, b, c = agents[0], agents[1], agents[2]
        # Pick random index between 0 to dimensions
        R = random.randint(0, self.population.genes)
        # Compute the agents potentially new position
        new_agent = Individual()
        for j in range(self.population.genes):
            # For each gene pick a uniformly distributed random number
            ri = round(random.uniform(0, 1), 2)
            if ri < self.crossover or i == R:
                new_agent.chromosome += [round(a.chromosome[j] + self.mutate * (b.chromosome[j] - c.chromosome[j]),
                                               self.population.precision)]
            else:
                new_agent.chromosome += [agent.chromosome[j]]

        new_agent.fitness = self.population.get_fitness(new_agent)
        if new_agent.fitness < agent.fitness:
            agent = new_agent
        return agent

    def three_unique_agents(self, agent):
        """
        Get 3 agents that are unique from each other and 'agent', from pop
        :param agent:
        :param pop:
        :return: list of 3 agents
        """
        agents = []
        while len(agents) < 3:
            i = random.randint(0, self.population.size-1)
            x = self.population.agents[i]
            if x != agent and x not in agents:
                agents += [x]
        return agents

    def run(self):
        """
        Run the DE Algorithm
        :return:
        """
        for generation in range(self.generations):
            new_population = []
            for i in range(self.population.size):
                new_population += [self.new_position(i)]
            self.population.agents = new_population
        print("Best: ", self.population.best_fitness())

if __name__ == "__main__":
    de = DifferentialEvolution()
    de.population = Population(size=100, genes=3, precision=6, domain=[0, pi])
    de.run()
