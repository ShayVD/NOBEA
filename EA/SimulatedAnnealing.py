import numpy as np
from NO.ComparativeBenchmarks import ComparativeBenchmarks
from By.Population import Population
import copy


class SimulatedAnnealing(Population):

    def __init__(self, max_temp, cool_rate, eval_limit, benchmark):
        """
        Uses metal annealing techniques to approximate global optimum of benchmark function.

        :param max_temp:
        :param cool_rate:
        :param eval_limit:
        :param benchmark:
        """
        super().__init__(size=1,eval_limit=eval_limit, benchmark=benchmark)
        self._max_temp = max_temp
        self._cool_rate = cool_rate
        self._temp = None
        self._step = 0

    @property
    def max_temp(self):
        return self._max_temp

    @max_temp.setter
    def max_temp(self, max_temp):
        self._max_temp = max_temp

    @property
    def cool_rate(self):
        return self._cool_rate

    @cool_rate.setter
    def cool_rate(self, cool_rate):
        self._cool_rate = cool_rate

    def annealing(self, precision=None, print_steps=True):
        """
        Run the simulated annealing algorithm.
        Temperature goes down each step.

        :param print_steps: bool, if True print state at each step
        :return:
        """
        if self.max_temp is None:
            self.get_max_temp()
        state = self.individuals[0]
        self._temp = self.max_temp
        generation = 0
        mins = []
        while self.evaluations < self.eval_limit:
            equilibrium = False
            while not equilibrium:
                solution = self.get_neighbour(state.solution)
                value = self.get_solutions_value(solution)
                fitness = self.get_fitness(value)
                if self.acceptance_probability(state.value, value, self._temp) > np.random.random():
                    equilibrium = True
                if self.evaluations % 100 == 0:
                    mins += [self.best_individual.value]
                if self.evaluations > self.eval_limit:
                    break
            state.solution = solution
            state.value = value
            state.fitness = fitness
            self.update_temperature()
            if state.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(state)
            if print_steps:
                print("Generation ", generation, "; Evaluations: ", self.evaluations,
                      "; Temperature: ", self._temp, "; ", state)
            if precision is not None and self.solution_precision(precision):
                break
            generation += 1
        return self.best_individual, mins

    def get_neighbour(self, sol):
        """
        Neighbouring solution to the one provided is calculated.

        :param sol: candidate solution vector
        :return:
        """
        std_dev = min(np.sqrt(self._temp), (self.domain[1]-self.domain[0])/3/self.cool_rate)
        solution = [None] * self.dimensions
        for i in range(self.dimensions):
            #rand = np.random.uniform(-np.pi/2, np.pi/2)
            #solution[i] = self.bind(sol[i] + self.cool_rate * self.T * np.tan(rand))
            solution[i] = self.bind(sol[i] + np.random.normal(0, 1) * std_dev * self.cool_rate)
        return solution

    def update_temperature(self):
        """
        Temperature is cooled.
        """
        self._step += 1
        # Would take a very long time
        # self.T = self.Tmax / np.log(self.k+1)
        # if self._temp <= 0.001:
        #    self._temp = 0.001
        #else:
        self._temp = self.max_temp / (1 + self._step)

    def get_max_temp(self):
        """
        Initial temperature is set to double the upper boundary.
        """
        self.max_temp = self.domain[1] * 2

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
    benchmark = ComparativeBenchmarks.f1()
    # benchmark.domain = [-10, 10]
    # benchmark.dimensions = 10
    sa = SimulatedAnnealing(max_temp=None, cool_rate=0.5, eval_limit=1000000, benchmark=benchmark)
    state, mins = sa.annealing(precision=2, print_steps=True)
    print("State: fitness=", state.fitness, "; value=", state.value)

