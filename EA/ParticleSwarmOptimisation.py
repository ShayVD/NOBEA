from By.Population import Population
from NO.ComparativeBenchmarks import ComparativeBenchmarks
from numpy import random


class ParticleSwarmOptimisation(Population):

    def __init__(self, inertia_weight, cognitive_constant, social_constant, size, generations, dimensions, domain,
                 precision, function):
        """
        Set class attributes.

        :param generations:         int, max iterations the algorithm should run for
        :param inertia_weight:      float, (0.0 -> 1.0)
        :param cognitive_constant:  int,
        :param social_constant:     int,
        """
        super().__init__(size=size, dimensions=dimensions, precision=precision, domain=domain, function=function)
        self.generations = generations
        self.inertia_weight = inertia_weight
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, generations):
        self._generations = generations

    @property
    def inertia_weight(self):
        return self._inertia_weight

    @inertia_weight.setter
    def inertia_weight(self, inertia_weight):
        self._inertia_weight = inertia_weight

    @property
    def cognitive_constant(self):
        return self._cognitive_constant

    @cognitive_constant.setter
    def cognitive_constant(self, cognitive_constant):
        self._cognitive_constant = cognitive_constant

    @property
    def social_constant(self):
        return self._social_constant

    @social_constant.setter
    def social_constant(self, social_constant):
        self._social_constant = social_constant

    def swarm(self, min_value=None, print_steps=True):
        """
        Swarm the population on the best solution.

        :return:    Individual, particle that produces the best solution
        """
        for generation in range(self.generations):
            for i in range(self.size):
                self.update_velocity(i)
                self.update_position(i)
                self.set_fitness(i)
            if print_steps:
                print("Generation:", generation+1, "/", self.generations, ";Solution:", self.best_individual)
            if min_value is not None and self.solution_precision(min_value):
                break
        return self.best_individual

    def update_velocity(self, index):
        """
        Update particle's velocity.

        :param particle:    Individual, particle to update it's velocity
        """
        particle = self.individuals[index]
        v = []
        for d in range(self.dimensions):
            rc = random.uniform(0, 1)
            rs = random.uniform(0, 1)
            cognitive = self.cognitive_constant * rc * (particle.best_solution[d] - particle.solution[d])
            social = self.social_constant * rs * (self.best_individual.solution[d] -
                                                  particle.solution[d])
            v += [self.inertia_weight * particle.velocity[d] + cognitive + social]
        particle.velocity = v

    def update_position(self, index):
        """
        Update particle's position within the search space.

        :param particle:    Individual, the particle to update it's position
        """
        particle = self.individuals[index]
        for d in range(self.dimensions):
            particle.solution[d] = self.bind(particle.solution[d] + particle.velocity[d])


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f1()
    pso = ParticleSwarmOptimisation(inertia_weight=0.2, cognitive_constant=1.9, social_constant=1.9, size=100,
                                    generations=3000, dimensions=benchmark.dimensions, domain=benchmark.domain,
                                    precision=2, function=benchmark.function)
    particle = pso.swarm(min_value=benchmark.min_value)
    print("Best Particle: ", particle)
