import numpy.random as random
from By.Population import Population
from NO.ComparativeBenchmarks import ComparativeBenchmarks


class GeneticAlgorithm(object):
    """
    GA is a meta-heuristic inspired by the process of natural selection.
    Relies on bio-inspired operators such as mutation, crossover, and selection.
    A population of candidate solutions, individuals, to an optimisation problem is evolved toward better solutions.
    Each individual has a set of properties, chromosomes, which can be mutated and altered.
    Evolution starts with a population of randomly generated individuals.
    In each generation the fitness of each individual is calculated using the fitness function.
    When a satisfactory solution is found to the fitness function, or max generations is reached, the algorithm stops.
    """

    def __init__(self, generations=100, crossovers=0.8, mutations=0.01, population=None):
        """
        Set class attributes.

        :param generations:     int, max number of iterations the algorithm's to run for
        :param crossovers:      float, (0.0 -> 1.0) percentage of population to create new solutions
        :param mutations:       float, (0.0 -> 1.0) percentage of population to have their chromosomes altered
        :param population:      Population, group of individuals to run the algorithm on
        """
        self._generations = generations
        self._crossovers = crossovers
        self._mutations = mutations
        self._population = population

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, generations):
        self._generations = generations

    @property
    def crossovers(self):
        return self._crossovers

    @crossovers.setter
    def crossovers(self, crossovers):
        self._crossovers = crossovers

    @property
    def mutations(self):
        return self._mutations

    @mutations.setter
    def mutations(self, mutations):
        self._mutations = mutations

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        self._population = population

    def evolve(self, min_value=None, print_steps=True):
        """
        Evolve the population using copy, crossover, and mutate.

        :param min_value:       float, algorithm terminates if solution with sufficient accuracy is found
        :param print_steps:     boolean, print solution for each generation if True
        :return:                individual, the best candidate solution found by the algorithm
        """
        for generation in range(self.generations):
            new_population = []
            # Copy
            new_population += self.copy()
            # Crossover
            new_population += self.crossover()
            # Mutate
            self.mutate(new_population)
            # Evaluate fitness of individuals
            self.population.individuals = new_population
            self.population.set_populations_fitness()
            if print_steps:
                print("Generation:", generation+1, "/", self.generations, ";Solution:", self.population.best_individual)
            if min_value is not None:
                if min_value < 0 and self.population.best_individual.value < min_value - (1*10**-self.population.precision):
                    break
                elif self.population.best_individual.value < min_value + (1 * 10 ** -self.population.precision):
                    break
        return self.population.best_individual

    def copy(self):
        """
        Copies the best individuals, picked by the selection function.
        Calculates amount of individuals to be copied for the next generation.
        Then uses a selection function to pick the individuals from the population.

        :return:    [individuals], individuals selected from population to survive for the next generation
        """
        amount_of_copies = round((1 - self.crossovers) * self.population.size)
        return self.rank_selection(amount_of_copies)

    def crossover(self):
        """
        Crosses over individuals to create offspring.
        Calculates amount of individuals, parents, that will be crossed over.
        Parents are then selected from the population using the selection function.
        Each pair of parents will create two offspring.
        **The amount of parents must be even as two parents are needed to create two children.

        :return:    [individuals], new individuals for the next generation
        """
        amount_of_parents = round(self.crossovers * self.population.size)
        parents = self.rank_selection(amount_of_parents)
        offspring = []
        for i in range(len(parents)):
            if i % 2 == 1:
                offspring_1, offspring_2 = self.population.create_individual(), self.population.create_individual()
                offspring_1.solution, offspring_2.solution = self.chromosome_crossover(parents[i - 1].solution,
                                                                                           parents[i].solution)
                offspring += [offspring_1, offspring_2]
        return offspring

    @staticmethod
    def chromosome_crossover(parent_1, parent_2):
        """
        Combines two parent chromosomes to create two new offspring chromosomes.

        :param parent_1:    [floats], chromosome of the first parent
        :param parent_2:    [floats], chromosome of the second parent
        :return:            [chromosomes], a list of the two new chromosomes
        """
        genes = len(parent_1)
        offspring_chromosomes = []
        for _ in range(2):
            new_chromosome = []
            for gene in range(genes):
                # TODO check if genes should be rounded to precision or just fitness
                new_chromosome += [random.uniform(min(parent_1[gene], parent_2[gene]),
                                                  max(parent_1[gene], parent_2[gene]))]
            offspring_chromosomes += [new_chromosome]
        return offspring_chromosomes

    def mutate(self, new_population):
        """
        Mutates random gene of random individuals in the population.
        Calculates the amount of individuals to have their chromosomes mutated.
        A mutation is randomly selected from anywhere in the search space.
        An individual is randomly selected, as is one of it's genes, and then the mutation becomes the selected gene.

        :param new_population:      [individuals], new population before mutation
        """
        amount_to_mutate = round(self.population.size * self.mutations)
        for _ in range(amount_to_mutate):
            mutation = random.uniform(self.population.domain[0], self.population.domain[1])
            rand_ind = random.randint(0, self.population.size)
            rand_gene = random.randint(0, self.population.dimensions)
            new_population[rand_ind].solution[rand_gene] = mutation

    def rank_selection(self, amount_of_copies):
        # Assign ranks and calculate total
        ranks = []
        total = 0
        for i in range(self.population.size):
            ranks += [1 / (i + 2)]
            total += 1 / (i + 2)
        # Assign probabilities based on rank
        probabilities = []
        for i in range(self.population.size):
            probabilities += [ranks[i]/total]
        # Select individuals based on probabilities
        self.population.sort_by_fitness()
        chosen = []
        for _ in range(amount_of_copies):
            selected = random.uniform(0, 1)
            for i in range(self.population.size):
                selected -= probabilities[i]
                if selected <= 0:
                    chosen += [self.population.individuals[i]]
                    break
        return chosen

    def roulette_wheel_selection(self, weighted_population):
        pass

    def tournament_selection(self):
        pass


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f1()
    population = Population(size=100, dimensions=3, precision=5, domain=benchmark.domain, function=benchmark.function)
    ga = GeneticAlgorithm(generations=250, crossovers=0.8, mutations=0.2, population=population)
    individual = ga.evolve(min_value=benchmark.min_value)
    print(individual)
