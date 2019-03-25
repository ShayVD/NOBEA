import numpy as np
from NO.ComparativeBenchmarks import ComparativeBenchmarks
from By.Population import Population
import copy


class SimulatedAnnealing(Population):

    def __init__(self, Tmax, Tmin, cool_rate, max_steps, size, dimensions, domain, precision, function):
        """

        :param max_steps:
        """
        super().__init__(size=size, dimensions=dimensions, precision=precision, domain=domain, function=function)
        self._max_steps = max_steps
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.cool_rate = cool_rate
        self.T = None
        self.step = 0

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
        if self.Tmax is None:
            self.get_max_temp()
        state = self.individuals[0]
        self.T = self.Tmax
        while self.T > self.Tmin:
            equilibrium = False
            while not equilibrium:
            #for i in range(5):
                solution = self.get_neighbour(state.solution)
                value = self.get_solutions_value(solution)
                fitness = self.get_fitness(value)
                if self.acceptance_probability(state.value, value, self.T) > np.random.random():
                    equilibrium = True
            state.solution = solution
            state.value = value
            state.fitness = fitness
            self.update_temperature()
            if state.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(state)
            if print_steps:
                print("Step ", self.step, "; Temperature: ", self.T, "; ", state)
            if min_value is not None and self.solution_precision(min_value):
                break
        return self.best_individual

    def get_neighbour(self, sol):
        std_dev = min(np.sqrt(self.T), (self.domain[1]-self.domain[0])/3/self.cool_rate)
        solution = [None] * self.dimensions
        for i in range(self.dimensions):
            #rand = np.random.uniform(-np.pi/2, np.pi/2)
            #solution[i] = self.bind(sol[i] + self.cool_rate * self.T * np.tan(rand))
            solution[i] = self.bind(sol[i] + np.random.normal(0, 1) * std_dev * self.cool_rate)
        return solution

    def update_temperature(self):
        self.step += 1
        # Would take a very long time
        # self.T = self.Tmax / np.log(self.k+1)
        self.T = self.Tmax / (1 + self.step)

    def get_max_temp(self):
        max_val = None
        min_val = None
        for i in range(50):
            state = self.create_individual()
            state.value = self.get_solutions_value(state.solution)
            state.fitness = self.get_fitness(state.value)
            if max_val is None or state.value > max_val:
                max_val = state.value
            if min_val is None or state.value < min_val:
                min_val = state.value
                self.best_individual = state
        self.Tmax = (max_val-min_val) * 1.5

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


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f17()
    sa = SimulatedAnnealing(Tmax=None, Tmin=0.1, cool_rate=0.5, max_steps=1000, size=1, dimensions=benchmark.dimensions, precision=2,
                            domain=benchmark.domain, function=benchmark.function)
    state = sa.annealing(print_steps=True, min_value=benchmark.min_value)
    print("State: ", state)
