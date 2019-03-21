import numpy as np
from NO.ComparativeBenchmarks import ComparativeBenchmarks
from By.Population import Population
import copy


class SimulatedAnnealing(Population):

    def __init__(self, max_steps, size, dimensions, domain, precision, function):
        """

        :param max_steps:
        """
        super().__init__(size=size, dimensions=dimensions, precision=precision, domain=domain, function=function)
        self._max_steps = max_steps

    @property
    def max_steps(self):
        return self._max_steps

    def annealing(self, min_value=None, print_steps=True):
        """
        Run the simulated annealing algorithm.
        Temperature goes down each step.

        :param print_steps: bool, if True print state at each step
        :return:
        """
        state = self.create_individual()
        state.value = self.get_solutions_value(state.solution)
        state.fitness = self.get_fitness(state.value)
        self.best_individual = copy.deepcopy(state)
        for step in range(self.max_steps):
            fraction = step / float(self.max_steps)
            T = self.temperature(fraction)
            new_state = self.create_individual()
            new_state.solution = self.neighbour(state.solution)
            new_state.value = self.get_solutions_value(new_state.solution)
            new_state.fitness = self.get_fitness(new_state.value)
            if new_state.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(new_state)
            if self.acceptance_probability(state.value, new_state.value, T) > np.random.random():
                state = new_state
            if print_steps:
                print("Step ", step, " / ", self.max_steps, "; ", state)
            if min_value is not None and self.solution_precision(min_value):
                break
        return self.best_individual

    def energy(self, x):
        """
        Result of x into the fitness function.

        :param x:   [floats], individuals chromosome
        :return:    float, individuals fitness
        """
        return self.function(x)

    def neighbour(self, x, fraction=1.0):
        """
        Get a random neighbour state, to the left, or right, of x, given the fraction (fraction=current_step/max_steps).
        Delta increases on average, per step.
        TODO See if this is correct.

        :param x:           [floats], individuals chromosome
        :param fraction:    float
        :return:            [floats], chromosome
        """
        amplitude = (self.domain[1] - self.domain[0]) * fraction / 10
        solution = [None] * self.dimensions
        for i in range(self.dimensions):
            delta = (-amplitude/2) + amplitude * np.random.random_sample()
            solution[i] = self.bind(x[i] + delta)
        return solution

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
    benchmark = ComparativeBenchmarks.f1()
    sa = SimulatedAnnealing(max_steps=50000, size=100, dimensions=1, precision=6, domain=benchmark.domain,
                            function=benchmark.function)
    state = sa.annealing(print_steps=True)
    print("State: ", state)
