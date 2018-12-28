from evolutionary_algorithms.Population import *
from math import *

class ParticleSwarmOptimisation(object):

    def __init__(self, generations=100, inertia_weight=0.5, cognative_constant=1, social_constant=2):
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
        self.inertia_weight = inertia_weight
        self.cognitive_constant = cognative_constant
        self.social_constant = social_constant
        self._population = None

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        self._population = population

    def update_velocity(self, particle):
        v = []
        for gene in range(self.population.genes):
            rc = random.uniform(0, 1)
            rs = random.uniform(0, 1)
            cognitive = self.cognitive_constant * rc * (particle.best_position[gene] - particle.chromosome[gene])
            social = self.social_constant * rs * (self.population.best_global_solution.chromosome[gene] - particle.chromosome[gene])
            v += [self.inertia_weight * particle.velocity[gene] + cognitive + social]
        particle.velocity = v

    def update_position(self, particle):
        for gene in range(self.population.genes):
            particle.chromosome[gene] = round(particle.chromosome[gene] + particle.velocity[gene], self.population.precision)
            if particle.chromosome[gene] < self.population.domain[0]:
                particle.chromosome[gene] = self.population.domain[0]
            elif particle.chromosome[gene] > self.population.domain[1]:
                particle.chromosome[gene] = self.population.domain[1]

    def run(self):
        """
        Run the DE Algorithm
        :return:
        """
        self.population.set_initial_velocity()
        self.population.set_initial_fitness()
        self.population.set_best_positions()
        self.population.set_best_global_solution()
        for generation in range(self.generations):
            for particle in self.population.agents:
                self.update_velocity(particle)
                self.update_position(particle)
                particle.fitness = self.population.get_fitness(particle)
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.chromosome
                if particle.best_fitness < self.population.best_global_solution.fitness:
                    self.population.best_global_solution = particle
        print("Best: ", self.population.best_fitness())

if __name__ == "__main__":
    pso = ParticleSwarmOptimisation(generations=100, inertia_weight=0.4, cognative_constant=1, social_constant=1)
    pso.population = Population(size=100, genes=3, precision=6, domain=[0, pi])
    pso.run()
