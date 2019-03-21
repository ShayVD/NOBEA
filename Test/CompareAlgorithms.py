from EA.GeneticAlgorithm import GeneticAlgorithm
from EA.DifferentialEvolution import DifferentialEvolution
from EA.ParticleSwarmOptimisation import ParticleSwarmOptimisation
from EA.ArtificialBeeColony import ArtificialBeeColony
from EA.SimulatedAnnealing import SimulatedAnnealing
from By.Population import Population
from NO.ComparativeBenchmarks import ComparativeBenchmarks
from Test.Spreadsheet import Spreadsheet
import copy


class CompareAlgorithms(object):

    @staticmethod
    def ga_parameter_tuning(benchmark, cr_jump=0.1, mt_jump=0.1, population_size=100, max_iterations=100):
        """
        Run GA on given benchmark problem, changing DE parameter values to find the best match.

        :param benchmark:
        :param cr_jump:
        :param mt_jump:
        :param population_size:
        :param max_iterations:
        """
        best_fitness = None
        best_params = (None, None)
        population = Population(size=population_size, dimensions=benchmark.dimensions, precision=None,
                                domain=benchmark.domain, function=benchmark.function)
        crossovers = 0.1
        while crossovers < 1.0:
            mutations = 0.1
            while mutations < 1.0:
                ga = GeneticAlgorithm(generations=max_iterations, crossovers=crossovers, mutations=mutations,
                                      population=copy.deepcopy(population))
                ind = ga.evolve(min_value=None, print_steps=False)
                if best_fitness is None or best_fitness < ind.fitness:
                    best_fitness = ind.fitness
                    best_params = (crossovers, mutations)
                print("Genetic Algorithm:      Fitness=", ind.fitness, "; Value=", ind.value)
                print("Crossovers: ", crossovers, "; Mutations: ", mutations)
                print("\n")
                mutations += mt_jump
            crossovers += cr_jump
        print("Params: ", best_params)
        print("Fitness: ", best_fitness)

    @staticmethod
    def de_parameter_tuning(benchmark, cr_jump=0.1, mt_jump=0.1, population_size=100, max_iterations=100):
        """
        Run DE on given benchmark problem, changing DE parameter values to find best match.

        :param benchmark:
        :param cr_jump:
        :param mt_jump:
        :param population_size:
        :param max_iterations:
        """
        best_fitness = None
        best_params = (None, None)
        population = Population(size=population_size, dimensions=benchmark.dimensions, precision=None,
                                domain=benchmark.domain, function=benchmark.function)
        crossovers = 0.1
        while crossovers < 1.0:
            mutations = 0.1
            while mutations < 1.0:
                de = DifferentialEvolution(generations=max_iterations, crossover=crossovers, mutate=mutations,
                                           population=copy.deepcopy(population))
                agent = de.evolve(min_value=None, print_steps=False)
                if best_fitness is None or best_fitness < agent.fitness:
                    best_fitness = agent.fitness
                    best_params = (crossovers, mutations)
                print("Differential Evolution:      Fitness=", agent.fitness, "; Value=", agent.value)
                print("Crossovers: ", crossovers, "; Mutations: ", mutations)
                print("\n")
                mutations += mt_jump
            crossovers += cr_jump
        print("Params: ", best_params)
        print("Fitness: ", best_fitness)

    @staticmethod
    def pso_parameter_tuning(benchmark, in_jump=0.1, cg_jump=0.1, sc_jump=0.1, population_size=100,
                             max_iterations=100):
        """
        Run PSO on given benchmark problem, changing PSO parameter values to find best match.

        :param benchmark:
        :param in_jump:
        :param cg_jump:
        :param sc_jump:
        :param population_size:
        :param max_iterations:
        """
        best_fitness = None
        best_params = (None, None, None)
        population = Population(size=population_size, dimensions=benchmark.dimensions, precision=None,
                                domain=benchmark.domain, function=benchmark.function)
        inertia = 0.1
        while inertia < 1.0:
            cognitive = 0.1
            while cognitive < 2.0:
                social = 0.1
                while social < 2.0:
                    pso = ParticleSwarmOptimisation(generations=max_iterations, inertia_weight=inertia,
                                                    cognitive_constant=cognitive, social_constant=social,
                                                    population=copy.deepcopy(population))
                    particle = pso.swarm(min_value=None, print_steps=False)
                    if best_fitness is None or best_fitness < particle.fitness:
                        best_fitness = particle.fitness
                        best_params = (inertia, cognitive, social)
                    print("PSO:           Fitness=", particle.fitness, "; Value=", particle.value)
                    print("Inertia: ", inertia, "; Cognitive: ", cognitive, "; Social: ", social)
                    print("\n")
                    social += sc_jump
                cognitive += cg_jump
            inertia += in_jump
        print("Params: ", best_params)
        print("Fitness: ", best_fitness)

    @staticmethod
    def abc_parameter_tuning(benchmark, lm_jump=1, population_size=100, max_iterations=100):
        """
        Run ABC on given benchmark problem, changing ABC parameter values to find best match.

        :param benchmark:
        :param lm_jump:
        :param population_size:
        :param max_iterations:
        """
        best_fitness = None
        best_params = None
        population = Population(size=population_size, dimensions=benchmark.dimensions, precision=None,
                                domain=benchmark.domain, function=benchmark.function)
        limit = 1
        while limit < 40:
            abc = ArtificialBeeColony(generations=max_iterations, limit=limit, population=copy.deepcopy(population))
            bee = abc.colonise(min_value=None, print_steps=False)
            if best_fitness is None or best_fitness < bee.fitness:
                best_fitness = bee.fitness
                best_params = limit
            print("Artificial Bee Colony:      Fitness=", bee.fitness, "; Value=", bee.value)
            print("Limit: ", limit)
            print("\n")
            limit += lm_jump
        print("Params: ", best_params)
        print("Fitness: ", best_fitness)



    @staticmethod
    def insert_result(algorithm, fitness, value, row):
        ss = Spreadsheet()
        ss.set_row_string(row, algorithm + ": Fitness= " + str(fitness) + " Value= " + str(value))

    @staticmethod
    def algorithms_all_benchmarks(population_size, generations):
        """
        Run each algorithm on every benchmark problem.
        **Very time consuming, should be used with low population size and generations

        :param population_size:
        :param generations:
        :return:
        """
        i = 1
        for benchmark in ComparativeBenchmarks.benchmarks_any_dimension():
            print("Benchmark ", i, " : ", benchmark.name, " - Minimum value: ", benchmark.min_value)

            population = Population(size=population_size, dimensions=benchmark.dimensions, precision=None,
                                    domain=benchmark.domain, function=benchmark.function)

            sa = SimulatedAnnealing(max_steps=generations)
            sa.population = copy.deepcopy(population)
            state = sa.annealing(min_value=None, print_steps=False)
            print("Simulated Annealing:         Fitness=", state.fitness, "; Value=", state.value)

            ga = GeneticAlgorithm(generations=generations, crossovers=0.9, mutations=0.1)
            ga.population = copy.deepcopy(population)
            individual = ga.evolve(min_value=None, print_steps=False)
            print("Genetic Algorithm:           Fitness=", individual.fitness, "; Value=", individual.value)

            de = DifferentialEvolution(generations=generations, crossover=0.8, mutate=0.2)
            de.population = copy.deepcopy(population)
            agent = de.evolve(min_value=None, print_steps=False)
            print("Differential Evolution:      Fitness=", agent.fitness, "; Value=", agent.value)

            pso = ParticleSwarmOptimisation(generations=generations, inertia_weight=0.2, cognitive_constant=1.9,
                                            social_constant=1.9)
            pso.population = copy.deepcopy(population)
            particle = pso.swarm(min_value=None, print_steps=False)
            print("Particle Swarm Optimisation: Fitness=", particle.fitness, "; Value=", particle.value)

            abc = ArtificialBeeColony(generations=generations, limit=20)
            abc.population = copy.deepcopy(population)
            bee = abc.colonise(min_value=None, print_steps=False)
            print("Artificial Bee Colony:       Fitness=", bee.fitness, "; Value=", bee.value)

            print("\n")
            i += 1

    @staticmethod
    def all_algorithms_benchmark_test(benchmark_number, population_size, generations, sa=True, ga=True, de=True,
                                      pso=True, abc=True):
        benchmark = ComparativeBenchmarks.benchmarks()[benchmark_number - 1]

        start_row = 8 * (benchmark_number - 1)
        ss = Spreadsheet()
        ss.set_row_string(start_row, "Population Size: " + str(population_size) + " Generations: " + str(generations))
        ss.set_row_string(start_row + 1,
                          "Benchmark: " + benchmark.name + " Minimum Value: " + str(benchmark.min_value) +
                          " Dimensions: " + str(10))

        if sa:
            sa = SimulatedAnnealing(max_steps=generations, size=population_size, dimensions=benchmark.dimensions,
                                    precision=None, domain=benchmark.domain, function=benchmark.function)
            state = sa.annealing(min_value=None, print_steps=True)
            CompareAlgorithms.insert_result("Simulated Annealing ", state.fitness, state.value, start_row + 2)
        if ga:
            ga = GeneticAlgorithm(generations=generations, crossovers=0.9, mutations=0.1, size=population_size,
                                  dimensions=benchmark.dimensions, precision=None, domain=benchmark.domain,
                                  function=benchmark.function)
            individual = ga.evolve(min_value=None, print_steps=True)
            CompareAlgorithms.insert_result("Genetic Algorithm ", individual.fitness, individual.value, start_row + 3)
        if de:
            de = DifferentialEvolution(generations=generations, crossover=0.8, mutate=0.2, size=population_size,
                                       dimensions=benchmark.dimensions, precision=None, domain=benchmark.domain,
                                       function=benchmark.function)
            agent = de.evolve(min_value=None, print_steps=True)
            CompareAlgorithms.insert_result("Differential Evolution ", agent.fitness, agent.value, start_row + 4)
        if pso:
            pso = ParticleSwarmOptimisation(generations=generations, inertia_weight=0.2, cognitive_constant=1.9,
                                            social_constant=1.9, size=population_size, dimensions=benchmark.dimensions,
                                            precision=None, domain=benchmark.domain, function=benchmark.function)
            particle = pso.swarm(min_value=None, print_steps=True)
            CompareAlgorithms.insert_result("Particle Swarm Optimisation", particle.fitness, particle.value,
                                            start_row + 5)
        if abc:
            abc = ArtificialBeeColony(generations=generations*5, employees=0.5, limit=20, size=20,
                                      dimensions=benchmark.dimensions, precision=None, domain=benchmark.domain,
                                      function=benchmark.function)
            bee = abc.colonise(min_value=None, print_steps=True)
            CompareAlgorithms.insert_result("Artificial Bee Colony", bee.fitness, bee.value, start_row + 6)


if __name__ == "__main__":
    ca = CompareAlgorithms()
    ca.all_algorithms_benchmark_test(benchmark_number=1, population_size=100, generations=100,
                                     sa=True, ga=True, de=True, pso=True, abc=True)
