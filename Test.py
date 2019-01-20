from EA.GeneticAlgorithm import GeneticAlgorithm
from EA.DifferentialEvolution import DifferentialEvolution
from EA.ParticleSwarmOptimisation import ParticleSwarmOptimisation
from EA.SimulatedAnnealing import SimulatedAnnealing
from EA.Population import Population
from NO.ComparativeBenchmarks import ComparativeBenchmarks

if __name__ == "__main__":
    i = 1
    for benchmark in ComparativeBenchmarks.benchmarks_any_dimension():
        print("Benchmark ", i, " : ", benchmark.name, " - Minimum value: ", benchmark.min_value)

        population = Population(size=100, genes=3, precision=6, domain=benchmark.domain,
                                fitness_function=benchmark.function)

        sa = SimulatedAnnealing(max_steps=1000)
        sa.population = population
        state = sa.annealing(print_steps=False)
        print("Simulated Annealing: ", state.fitness)

        population.reset()

        ga = GeneticAlgorithm(generations=100, crossovers=0.8, mutations=0.1)
        ga.population = population
        individual = ga.evolve(min_value=benchmark.min_value, print_steps=False)
        print("Genetic Algorithm: ", individual.fitness)

        population.reset()

        de = DifferentialEvolution(generations=100, crossover=0.9, mutate=0.8)
        de.population = population
        agent = de.evolve(min_value=benchmark.min_value, print_steps=False)
        print("Differential Evolution: ", agent.fitness)

        population.reset()

        pso = ParticleSwarmOptimisation(generations=100, inertia_weight=0.4, cognitive_constant=1, social_constant=1)
        pso.population = population
        particle = pso.swarm(min_value=benchmark.min_value, print_steps=False)
        print("Particle Swarm Optimisation: ", particle.fitness)

        print("\n")
        i += 1
