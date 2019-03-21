import numpy.random as random
from By.Population import Population
from NO.ComparativeBenchmarks import ComparativeBenchmarks


class GeneticAlgorithm(Population):
    """
    GA is a meta-heuristic inspired by the process of natural selection.
    Relies on bio-inspired operators such as mutation, crossover, and selection.
    A population of candidate solutions, individuals, to an optimisation problem is evolved toward better solutions.
    Each individual has a set of properties, chromosomes, which can be mutated and altered.
    Evolution starts with a population of randomly generated individuals.
    In each generation the fitness of each individual is calculated using the fitness function.
    When a satisfactory solution is found to the fitness function, or max generations is reached, the algorithm stops.
    """

    def __init__(self, crossovers, mutations, size, generations, dimensions, domain, precision, function):
        """
        Set class attributes.

        :param generations:     int, max number of iterations the algorithm's to run for
        :param crossovers:      float, (0.0 -> 1.0) percentage of population to create new solutions
        :param mutations:       float, (0.0 -> 1.0) percentage of population to have their chromosomes altered
        """
        super().__init__(size=size, dimensions=dimensions, precision=precision, domain=domain, function=function)
        self._generations = generations
        self._crossovers = crossovers
        self._mutations = mutations

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
            self.individuals = new_population
            self.set_populations_fitness()
            if print_steps:
                print("Generation:", generation+1, "/", self.generations, ";Solution:", self.best_individual)
            if min_value is not None and self.solution_precision(min_value):
                break
        return self.best_individual

    def copy(self):
        """
        Copies the best individuals, picked by the selection function.
        Calculates amount of individuals to be copied for the next generation.
        Then uses a selection function to pick the individuals from the population.

        :return:    [individuals], individuals selected from population to survive for the next generation
        """
        amount_of_copies = round((1 - self.crossovers) * self.size)
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
        amount_of_parents = round(self.crossovers * self.size)
        parents = self.rank_selection(amount_of_parents)
        offspring = []
        for i in range(len(parents)):
            if i % 2 == 1:
                offspring_1, offspring_2 = self.create_individual(), self.create_individual()
                offspring_1.solution, offspring_2.solution = self._chromosome_crossover(parents[i - 1].solution,
                                                                                           parents[i].solution)
                offspring += [offspring_1, offspring_2]
        return offspring

    def mutate(self, new_population):
        """
        Mutates random gene of random individuals in the population.
        Calculates the amount of individuals to have their chromosomes mutated.
        A mutation is randomly selected from anywhere in the search space.
        An individual is randomly selected, as is one of it's genes, and then the mutation becomes the selected gene.

        :param new_population:      [individuals], new population before mutation
        """
        amount_to_mutate = round(self.size * self.mutations)
        for _ in range(amount_to_mutate):
            mutation = random.uniform(self.domain[0], self.domain[1])
            rand_ind = random.randint(0, self.size)
            rand_gene = random.randint(0, self.dimensions)
            new_population[rand_ind].solution[rand_gene] = mutation

    @staticmethod
    def _chromosome_crossover(parent_1, parent_2):
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
                new_chromosome += [random.uniform(min(parent_1[gene], parent_2[gene]),
                                                  max(parent_1[gene], parent_2[gene]))]
            offspring_chromosomes += [new_chromosome]
        return offspring_chromosomes

    """/--------------------------------------SELECTION METHOD------------------------------------------------------/"""

    def rank_selection(self, amount_of_copies):
        self.sort_by_fitness(increasing_order=False)
        probabilities = self._rank_selection_probabilities()
        chosen = []
        for _ in range(amount_of_copies):
            index = self._rank_selection_index(probabilities)
            chosen += [self.individuals[index]]
        return chosen

    def _rank_selection_index(self, probabilities):
        selected = random.uniform(0, 1)
        for i in range(self.size):
            selected -= probabilities[i]
            if selected <= 0:
                return i

    def _rank_selection_probabilities(self):
        ranks = []
        total = 0
        for i in range(self.size):
            ranks += [1 / (i + 2)]
            total += 1 / (i + 2)
        probabilities = []
        for i in range(self.size):
            probabilities += [ranks[i] / total]
        return probabilities

    """/--------------------------------------OTHER SELECTION METHODS-----------------------------------------------/"""

    def tournament_selection(self, amount):
        tourny_size = self.size // 10
        chosen = []
        for i in range(amount):
            index = self._tournament_selection_index(tourny_size)
            chosen += [self.individuals[index]]
        return chosen

    def _tournament_selection_index(self, tourny_size):
        best = None
        best_index = None
        for i in range(tourny_size):
            index = random.randint(0, self.size)
            ind = self.individuals[index]
            if best is None or ind.fitness < best.fitness:
                best = ind
                best_index = index
        return best_index

    def roulette_wheel_selection(self, amount):
        chosen = []
        for _ in range(amount):
            random.shuffle(self.individuals)
            index = self._roulette_wheel_index()
            chosen += [self.individuals[index]]
        return chosen

    def _roulette_wheel_index(self):
        probabilities = self._roulette_wheel_probabilities()
        beta = random.uniform(0, max(probabilities))
        current = 0
        for i in range(self.size):
            current += probabilities[i]
            if current > beta:
                return i

    def _roulette_wheel_probabilities(self):
        total = 0
        for j in self.individuals:
            total += j.fitness
        probabilities = []
        for i in self.individuals:
            probabilities += [i.fitness/total]
        return probabilities


if __name__ == "__main__":
    benchmark = ComparativeBenchmarks.f1()
    ga = GeneticAlgorithm(crossovers=0.9, mutations=0.1, size=500, generations=100, dimensions=3,
                          domain=benchmark.domain, precision=6, function=benchmark.function)
    individual = ga.evolve(min_value=None, print_steps=True)
    print(individual)
