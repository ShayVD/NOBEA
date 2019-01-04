import numpy as np
from math import *
from EA.Population import *
from EA.Individual import *
import NO.comparativeBenchmarks as cb


class GeneticAlgorithm(object):

    def __init__(self, generations=100, crossover=0.8, mutate=0.01):
        """


        :param generations:
        :param crossover:
        :param mutate:
        """
        self._generations = generations
        self._crossover = crossover
        self._mutate = mutate
        self._population = None

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, generations):
        self._generations = generations

    @property
    def crossover(self):
        return self._crossover

    @crossover.setter
    def population(self, crossover):
        self._crossover = crossover

    @property
    def mutate(self):
        return self._mutate

    @mutate.setter
    def mutate(self, mutate):
        self._mutate = mutate

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        self._population = population

    """def best_selection(self, weighted_population):
        amount_of_copies = round((1 - self.crossover) * self.population.size)
        return weighted_population[0:amount_of_copies]"""

    def roulette_wheel_selection(self, weighted_population):
        """
        Select which individuals to be copied to the next generation using roulette wheel selection.
        Probability of selection is equal to relative fitness.

        :return: list of individuals to be added to next generation
        """
        amount_of_copies = round((1 - self.crossover) * self.population.size)
        chosen = []
        total_fitness = 0
        probabilities = []
        offset = weighted_population[0][1]

        # Calculate sum of all individuals fitness'
        for ind in weighted_population:
            total_fitness += ind.fitness - offset + 1
        print("total fitness: ", total_fitness)

        total = 0
        # Calculate individuals probability of being picked
        for ind in weighted_population:
            total += (ind.fitness - offset + 1)/total_fitness
            probabilities += [(ind.fitness - offset + 1)/total_fitness]
        print("probabilities: ", probabilities)
        print(total)

        # Pick individuals based on probability
        for i in range(amount_of_copies):
            selected = random.uniform
        print(chosen)
        return chosen

    def rank_selection(self, amount_of_copies):
        # Assign ranks and calculate total
        ranks = []
        total = 0
        for i in range(self.population.size):
            ranks += [1 / (i + 2)]
            total += 1 / (i+2)
        # Assign probabilities based on rank
        probabilities = []
        for i in range(self.population.size):
            probabilities += [ranks[i]/total]
        # Select individuals based on probabilities
        chosen = []
        for _ in range(amount_of_copies):
            selected = random.uniform(0, 1)
            for i in range(self.population.size):
                selected -= probabilities[i]
                if selected <= 0:
                    chosen += [self.population.individuals[i]]
                    break
        return chosen

    def tournament_selection(self):
        pass

    def ga_mutate(self, population):
        amount_to_mutate = round(self.population.size * self.mutate)
        for _ in range(amount_to_mutate):
            mutation = round(random.uniform(self.population.domain[0], self.population.domain[1]),
                             self.population.precision)
            rand_ind = np.random.randint(0, self.population.size)
            rand_gene = np.random.randint(0, self.population.genes)
            population[rand_ind].chromosome[rand_gene] = mutation

    def ga_copy(self):
        amount_of_copies = round((1 - self.crossover) * self.population.size)
        return self.rank_selection(amount_of_copies)

    def ga_crossover(self):
        offspring = []
        amount_of_parents = round(self.crossover * self.population.size)
        parents = self.rank_selection(amount_of_parents)
        # Length of parents must be even
        for i in range(len(parents)):
            if i % 2 == 1:
                offspring += self.real_crossover(parents[i - 1], parents[i])
        return offspring

    def real_crossover(self, parent1, parent2):
        offspring = []
        for _ in range(2):
            ind = Individual(self.population.genes, self.population.precision, self.population.domain)
            for gene in range(self.population.genes):
                ind.chromosome[gene] = round(random.uniform(min(parent1.chromosome[gene], parent2.chromosome[gene]),
                                                            max(parent1.chromosome[gene], parent2.chromosome[gene])
                                                            ), self.population.precision)
                ind.fitness = self.population.get_fitness(ind)
            offspring += [ind]
        return offspring

    def run(self):
        """
        Run the Genetic Algorithm.

        :return:
        """
        #Rank selection only works with sorted population
        self.population.sort_by_fitness()
        for generation in range(self.generations):
            new_population = []
            # Copy
            new_population += self.ga_copy()
            # Crossover
            new_population += self.ga_crossover()
            # Mutate
            self.ga_mutate(new_population)
            # Evaluate fitness of individuals
            self.population.individuals = new_population
            self.population.set_population_fitness()
            self.population.sort_by_fitness()
        print("Genetic Algorithm Best: ", self.population.best_individual)


if __name__ == "__main__":
    ga = GeneticAlgorithm(generations=100, crossover=0.8, mutate=0.01)
    ga.population = Population(size=100, genes=2, precision=6, domain=[-50, 50], fitness_function=cb.f13)
    ga.run()
