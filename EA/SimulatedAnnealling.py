"""
State space = domain
Energy (goal) function E()
Candidate generator procedure neighbour()
Acceptance probability function P()
Annealing schedule temperature()
Initial temperature

Pseudo-code:
Let s = s0
For k = 0 through kMax(exclusive):
    T <- temperature(k/kMax)
    Pick a random neighbour, sNew <- neighbour(s)
    If P( E(s), E(sNew), T ) >= random(0,1):
        s <- sNew
Output: final state s
"""
import numpy as np
import NO.comparativeBenchmarks as cb
from EA.Population import Population


class SimulatedAnnealing(object):

    def __init__(self, max_steps):
        """

        :param max_steps:
        """
        self._max_steps = max_steps
        self._population = None

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        self._population = population

    def annealing(self, print_steps=True):
        """
        Run the simulated annealing algorithm.
        Temperature goes down each step.

        :param print_steps: bool, if True print state at each step
        :return:
        """
        state = self.population.create_individual()
        state.set_fitness(self.energy(state.chromosome))
        for step in range(self.max_steps):
            fraction = step / float(self.max_steps)
            T = self.temperature(fraction)
            new_state = self.population.create_individual()
            new_state.chromosome = self.neighbour(state.chromosome)
            new_state.set_fitness(self.energy(new_state.chromosome))
            if self.acceptance_probability(state.fitness, new_state.fitness, T) > np.random.random():
                state = new_state
            if print_steps:
                print("Step ", step, " / ", self.max_steps, "; ", state)
            if state.fitness < 1*10**-self.population.precision:
                break
        return state

    def bind(self, x):
        """
        Keeps x within the domain.

        :param x:   float
        :return:    float
        """
        if x < self.population.domain[0]:
            return self.population.domain[0]
        elif x > self.population.domain[1]:
            return self.population.domain[1]
        else:
            return x

    def energy(self, x):
        """
        Result of x into the fitness function.

        :param x:   [floats], individuals chromosome
        :return:    float, individuals fitness
        """
        return self.population.fitness_function(x)

    def neighbour(self, x, fraction=1.0):
        """
        Get a random neighbour state, to the left, or right, of x, given the fraction (fraction=current_step/max_steps).
        Delta increases on average, per step.
        TODO See if this is correct.

        :param x:           [floats], individuals chromosome
        :param fraction:    float
        :return:            [floats], chromosome
        """
        amplitude = (self.population.domain[1] - self.population.domain[0]) * fraction / 10
        chromosome = [None] * self.population.genes
        for i in range(self.population.genes):
            delta = (-amplitude/2) + amplitude * np.random.random_sample()
            chromosome[i] = self.bind(x[i] + delta)
        return chromosome

    @staticmethod
    def acceptance_probability(cost, new_cost, temperature):
        """
        Probability that new state will be accepted.
        If new cost is less than current cost then probability is 100%.
        Otherwise the probability is based on the current cost, new cost, and temperature.

        :param cost:            float, fitness of current solution
        :param new_cost:        float, fitness of new solution
        :param temperature:     float, current temperature
        :return:                float, acceptance probability, between 0 and 1
        """
        if new_cost < cost:
            return 1
        return np.exp(- (new_cost - cost) / temperature)

    @staticmethod
    def temperature(fraction):
        """
        Temperature based on fraction, fraction = current step / max steps.

        :param fraction:
        :return:
        """
        """ Example of temperature decreasing as the process goes on."""
        if fraction < 0.01:
            return 0.01
        elif fraction > 1:
            return 1
        else:
            return 1 - fraction


if __name__ == "__main__":
    sa = SimulatedAnnealing(max_steps=1000)
    sa.population = Population(size=100, genes=1, precision=6, domain=[-10, 10], fitness_function=cb.f1)
    state = sa.annealing(print_steps=True)
    print("State: ", state)
