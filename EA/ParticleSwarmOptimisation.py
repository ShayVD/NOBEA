from By.Population import Population
from NO.ComparativeBenchmarks import ComparativeBenchmarks
from numpy import random


class ParticleSwarmOptimisation(Population):

    def __init__(self, inertia_weight, cognitive_constant, social_constant, size, eval_limit, benchmark):
        """
        Uses animal swarming techniques to approximate global optimum of benchmark function.

        :param inertia_weight:      float, (0.0 -> 1.0)
        :param cognitive_constant:  int,
        :param social_constant:     int,
        """
        super().__init__(size=size, eval_limit=eval_limit, benchmark=benchmark)
        self._inertia_weight = inertia_weight
        self._cognitive_constant = cognitive_constant
        self._social_constant = social_constant

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

    def swarm(self, precision=None, print_steps=True):
        """
        Swarm the population on the best solution.

        :return:    Individual, particle that produces the best solution
        """
        generation = 0
        mins = []
        while self.evaluations < self.eval_limit:
            for i in range(self.size):
                self.update_velocity(i)
                self.update_position(i)
                self.set_fitness(i)
            if print_steps:
                print("Generation:", generation+1, "; Evaluations: ", self.evaluations,
                      ";Solution:", self.best_individual)
            if precision is not None and self.solution_precision(precision):
                break
            generation += 1
            mins += [self.best_individual.value]
        return self.best_individual, mins

    def update_velocity(self, index):
        """
        Update particle's velocity.

        :param index:    particles index in population list
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

        :param index:    Individual, the particle to update it's position
        """
        particle = self.individuals[index]
        for d in range(self.dimensions):
            particle.solution[d] = self.bind(particle.solution[d] + particle.velocity[d])


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f2()
    # benchmark.domain = [-10, 10]
    # benchmark.dimensions = 10
    pso = ParticleSwarmOptimisation(inertia_weight=0.6, cognitive_constant=1.8, social_constant=1.8, size=20,
                                    eval_limit=100000, benchmark=benchmark)
    particle, mins = pso.swarm(precision=None, print_steps=True)
    print("Best Particle: ", particle)
