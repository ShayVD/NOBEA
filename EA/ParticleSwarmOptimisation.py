from EA.Population import Population
from NO.ComparativeBenchmarks import ComparativeBenchmarks
from numpy import random


class ParticleSwarmOptimisation(object):

    def __init__(self, generations=100, inertia_weight=0.5, cognitive_constant=1, social_constant=2, population=None):
        """
        Set class attributes.

        :param generations:         int, max iterations the algorithm should run for
        :param inertia_weight:      float, (0.0 -> 1.0)
        :param cognitive_constant:  int,
        :param social_constant:     int,
        :param population:          Population, group of particles to run the algorithm on
        """
        self.generations = generations
        self.inertia_weight = inertia_weight
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self._population = population

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        self._population = population

    def swarm(self, min_value=None, print_steps=True):
        """
        Swarm the population on the best solution.

        :return:    Individual, particle that produces the best solution
        """
        for generation in range(self.generations):
            for particle in self.population.individuals:
                self.update_velocity(particle)
                self.update_position(particle)
                self.population.set_individual_fitness(particle)
            if print_steps:
                print("Generation:", generation+1, "/", self.generations, ";Solution:", self.population.best_individual)
            if min_value is not None:
                if min_value < 0 and self.population.best_individual.fitness<min_value-1*10**-self.population.precision:
                    break
                elif self.population.best_individual.fitness < min_value+1*10**-self.population.precision:
                    break
        return self.population.best_individual

    def update_velocity(self, particle):
        """
        Update particle's velocity.

        :param particle:    Individual, particle to update it's velocity
        """
        v = []
        for gene in range(self.population.genes):
            rc = random.uniform(0, 1)
            rs = random.uniform(0, 1)
            cognitive = self.cognitive_constant * rc * (particle.best_chromosome[gene] - particle.chromosome[gene])
            social = self.social_constant * rs * (self.population.best_individual.chromosome[gene] -
                                                  particle.chromosome[gene])
            v += [self.inertia_weight * particle.velocity[gene] + cognitive + social]
        particle.velocity = v

    def update_position(self, particle):
        """
        Update particle's position within the search space.

        :param particle:    Individual, the particle to update it's position
        """
        for gene in range(self.population.genes):
            particle.chromosome[gene] = round(particle.chromosome[gene] + particle.velocity[gene],
                                              self.population.precision)
            if particle.chromosome[gene] < self.population.domain[0]:
                particle.chromosome[gene] = self.population.domain[0]
            elif particle.chromosome[gene] > self.population.domain[1]:
                particle.chromosome[gene] = self.population.domain[1]


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f1()
    population = Population(size=100, genes=3, precision=6, domain=benchmark.domain,
                            fitness_function=benchmark.function)
    pso = ParticleSwarmOptimisation(generations=100, inertia_weight=0.4, cognitive_constant=1, social_constant=1,
                                    population=population)
    particle = pso.swarm(benchmark.min_value)
    print("Best Particle: ", particle)
