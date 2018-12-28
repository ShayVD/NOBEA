import numpy as np
from math import *
import random

class GA_Simple(object):

    def __init__(self, precision=6, population_size=50, domain=[-1, 2], generations=100,
                 crossover=0.8, mutate=0.1):
        self.precision = precision
        self.population_size = population_size
        self.generations = generations
        self.crossover = crossover
        self.mutate = mutate
        self.domain = domain
        self.domain_length = abs(-self.domain[1] + self.domain[0])
        self.bits_length = int.bit_length(self.domain_length * 1 * 10 ** self.precision)

    def random_individual(self):
        """
        Creates an initial random individual of the population
        :return: a list of size 'dna'
        """
        x = []
        for i in range(self.bits_length):
            x += [np.random.randint(0, 2)]
        x = np.asarray(x)
        return x

    def initial_population(self):
        """
        Creates the initial random population
        :return: a list of length 'population' individuals
        """
        population = []
        for i in range(self.population_size):
            population += [self.random_individual()]
        return population

    def convert_bits_to_real_number(self, bits):
        """
        Convert binary representation of solution to a real number
        :param bits: binary representation of the solution
        :return: int
        """
        bit_string = ""
        for bit in bits:
            bit_string += str(bit)
        return round(self.domain[0] + int(bit_string, 2) * (self.domain_length/((2 ** self.bits_length) - 1)), self.precision)

    def fitness(self, individual):
        """
        Get fitness of individual.
        :param individual:
        :return:
        """
        real_number = self.convert_bits_to_real_number(individual)
        return round(real_number * sin(10 * pi * real_number) + 1.0, self.precision)

    def fitness_f(self, f):
        pass

    def average_fitness(self):
        pass

    def best_selection(self, weighted_population):
        amount_of_copies = round((1 - self.crossover) * self.population_size)
        return weighted_population[0:amount_of_copies]

    def roulette_wheel_selection(self, weighted_population):
        """
        Select which individuals to be copied to the next generation
        using roulette wheel selection. Probability of selection is equal
        to relative fitness.
        :return: list of individuals to be added to next generation
        """
        amount_of_copies = round((1 - self.crossover) * self.population_size)
        chosen = []
        total_fitness = 0
        probabilities = []
        offset = weighted_population[0][1]

        # Calculate sum of all individuals fitness'
        for ind in weighted_population:
            total_fitness += ind[1] - offset + 1
        print("total fitness: ", total_fitness)

        total = 0
        # Calculate individuals probability of being picked
        for ind in weighted_population:
            total += (ind[1] - offset + 1)/total_fitness
            probabilities += [(ind[1] - offset + 1)/total_fitness]
        print("probabilities: ", probabilities)
        print(total)

        # Pick individuals based on probability
        for i in range(amount_of_copies):
            selected = random.uniform
        print(chosen)
        return chosen

    def rank_selection(self, weighted_population, amount_of_copies):

        # Assign ranks and calculate total
        ranks = []
        total = 0
        for i in range(len(weighted_population)):
            ranks += [1 / (i + 2)]
            total += 1 / (i+2)

        # Assign probabilities based on rank
        probabilities = []
        for i in range(len(weighted_population)):
            probabilities += [ranks[i]/total]

        # Select individuals based on probabilities
        chosen = []
        for _ in range(amount_of_copies):
            selected = random.uniform(0, 1)
            for i in range(self.population_size):
                selected -= probabilities[i]
                if selected <= 0:
                    chosen += [weighted_population[i]]
                    break
        return chosen

    def tournament_selection(self):
        pass

    def single_point_crossover(self, parent1, parent2):
        mask1 = []
        mask2 = []
        for i in range(self.bits_length):
            if i % 2 == 0:
                mask1 += [1]
                mask2 += [0]
            else:
                mask1 += [0]
                mask2 += [1]
        mask1 = np.asarray(mask1)
        mask2 = np.asarray(mask2)
        offspring1 = np.bitwise_or(np.bitwise_and(parent1, mask1), np.bitwise_and(parent2, mask2))
        offspring2 = np.bitwise_or(np.bitwise_and(parent1, mask2), np.bitwise_and(parent2, mask1))
        return [(offspring1, 0), (offspring2, 0)]

    def run(self):
        """
        Run the Genetic Algorithm
        :return:
        """
        # Create initial population
        population = self.initial_population()

        # Compute fitness of individuals
        weighted_pop = []
        for individual in population:
            fitness = self.fitness(individual)
            weighted_pop += [(individual, fitness)]

        # Sort population so that the individual with the min value is first
        weighted_pop.sort(key=lambda x: x[1])
        print(weighted_pop)
        print(self.convert_bits_to_real_number(weighted_pop[0][0]))

        # Do all generations, or, if max fitness is reached
        for generation in range(self.generations):
            new_population = []

            # Copy; Roulette Wheel, Rank, Tournament Selection
            copies = round((1 - self.crossover) * self.population_size)
            new_population += self.rank_selection(weighted_pop, copies)

            # Crossover
            copies = round(self.crossover * self.population_size)
            parents = self.rank_selection(weighted_pop, copies)
            offspring = []
            # Length of parents must be even
            for i in range(len(parents)):
                if i % 2 == 1:
                    offspring += self.single_point_crossover(parents[i-1][0], parents[i][0])
            new_population += offspring


            # Mutate offspring
            # Create random mask with 1 different bit
            mutation = []
            bit_index = np.random.randint(0, self.bits_length)
            for i in range(self.bits_length):
                if i == bit_index:
                    mutation += [1]
                else:
                    mutation += [0]
            mutation = np.asarray(mutation)

            amount_to_mutate = round(self.population_size * self.mutate)
            for _ in range(amount_to_mutate):
                i = np.random.randint(0, self.population_size)
                new_population[i] = (np.bitwise_xor(new_population[i][0], mutation), 0)


            # Evaluate fitness of individuals
            new_population.sort(key=lambda x: x[1])
            population = []
            for individual in new_population:
                fitness = self.fitness(individual[0])
                population += [(individual[0], fitness)]
        population.sort(key=lambda x: x[1])
        print(population)
        print(self.convert_bits_to_real_number(population[0][0]))


if __name__ == "__main__":
    ga = GA_Simple()
    ga.run()
