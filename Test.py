from EA.GeneticAlgorithm import GeneticAlgorithm
from EA.DifferentialEvolution import DifferentialEvolution
from EA.ParticleSwarmOptimisation import ParticleSwarmOptimisation
from EA.ArtificialBeeColony import ArtificialBeeColony
from EA.SimulatedAnnealing import SimulatedAnnealing
from By.Population import Population
from NO.ComparativeBenchmarks import ComparativeBenchmarks
import copy


class CompareAlgorithms(object):

    def ga_benchmark1(self):
        benchmark = ComparativeBenchmarks.f1()
        population = Population(size=100, dimensions=10, precision=6, domain=benchmark.domain,
                                function=benchmark.function)
        crossovers = 0.0
        while crossovers != 1.1:
            crossovers += 0.1
            mutations = 0.0
            while mutations < 1.0:
                ga = GeneticAlgorithm(generations=250, crossovers=crossovers, mutations=mutations,
                                      population=copy.deepcopy(population))
                individual = ga.evolve(min_value=benchmark.min_value, print_steps=False)
                print("Genetic Algorithm:           Fitness=", individual.fitness, "; Value=", individual.value)
                print("Crossovers: ", crossovers, "; Mutations: ", mutations)
                mutations += 0.1


if __name__ == "__main__":
    ca = CompareAlgorithms()
    ca.ga_benchmark1()
    """i = 1
    for benchmark in ComparativeBenchmarks.benchmarks_any_dimension():
        print("Benchmark ", i, " : ", benchmark.name, " - Minimum value: ", benchmark.min_value)

        population = Population(size=100, dimensions=10, precision=6, domain=benchmark.domain,
                                function=benchmark.function)

        sa = SimulatedAnnealing(max_steps=1000)
        sa.population = copy.deepcopy(population)
        state = sa.annealing(min_value=benchmark.min_value, print_steps=False)
        print("Simulated Annealing:         Fitness=", state.fitness, "; Value=", state.value)

        ga = GeneticAlgorithm(generations=250, crossovers=0.8, mutations=0.1)
        ga.population = copy.deepcopy(population)
        individual = ga.evolve(min_value=benchmark.min_value, print_steps=False)
        print("Genetic Algorithm:           Fitness=", individual.fitness, "; Value=", individual.value)

        de = DifferentialEvolution(generations=250, crossover=0.9, mutate=0.8)
        de.population = copy.deepcopy(population)
        agent = de.evolve(min_value=benchmark.min_value, print_steps=False)
        print("Differential Evolution:      Fitness=", agent.fitness, "; Value=", agent.value)

        pso = ParticleSwarmOptimisation(generations=250, inertia_weight=0.4, cognitive_constant=1, social_constant=1)
        pso.population = copy.deepcopy(population)
        particle = pso.swarm(min_value=benchmark.min_value, print_steps=False)
        print("Particle Swarm Optimisation: Fitness=", particle.fitness, "; Value=", particle.value)

        abc = ArtificialBeeColony(generations=250, limit=20)
        abc.population = copy.deepcopy(population)
        bee = abc.colonise(min_value=benchmark.min_value, print_steps=False)
        print("Artificial Bee Colony:       Fitness=", bee.fitness, "; Value=", bee.value)

        print("\n")
        i += 1"""

