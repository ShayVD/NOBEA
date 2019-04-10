from EA.GeneticAlgorithm import GeneticAlgorithm
from EA.DifferentialEvolution import DifferentialEvolution
from EA.ParticleSwarmOptimisation import ParticleSwarmOptimisation
from EA.ArtificialBeeColony import ArtificialBeeColony
from EA.SimulatedAnnealing import SimulatedAnnealing
from NO.ComparativeBenchmarks import ComparativeBenchmarks


class CompareAlgorithms(object):

    def csv(self, results, file):
        """
        Write results to a file.

        :param results: list of values
        :param file: filename
        """
        with open(file, "w") as tf:
            min_len = min(results, key=lambda i: len(i))
            for j in range(len(min_len)):
                for k in range(len(results)):
                    tf.write(str(results[k][j])+",")

                tf.write("\n")

    def ABC(self, eval_limit=100000, employees=0.5, limit=100, size=20, benchmarks=(ComparativeBenchmarks.f1()),
            precision=None):
        """
        Run ABC algorithm on list of benchmarks. 30 executions per benchmark.

        :param eval_limit:
        :param employees:
        :param limit:
        :param size:
        :param benchmarks:
        :param precision:
        """
        for i in range(len(benchmarks)):
            benchmark = benchmarks[i]
            abc = ArtificialBeeColony(employees=employees, limit=limit, size=size, eval_limit=eval_limit,
                                      benchmark=benchmark)
            results = []
            for j in range(30):
                bee, mins = abc.colonise(precision=precision, print_steps=False)
                results += [mins]
                abc.__init__(employees=employees, limit=limit, size=size, eval_limit=eval_limit, benchmark=benchmark)
                print(i+13, j+1, bee.value)
            self.csv(results, "ABC_Results/abc_output_"+str(i+13)+".csv")

    def DE(self, eval_limit=100000, crossover=0.8, mutate=0.2, size=100, benchmarks=(ComparativeBenchmarks.f1()),
           precision=None):
        """
        Run DE algorithm on list of benchmarks. 30 executions per benchmark.

        :param eval_limit:
        :param crossover:
        :param mutate:
        :param size:
        :param benchmarks:
        :param precision:
        """
        for i in range(len(benchmarks)):
            benchmark = benchmarks[i]
            de = DifferentialEvolution(crossover=crossover, mutate=mutate, size=size, eval_limit=eval_limit,
                                       benchmark=benchmark)
            results = []
            for j in range(30):
                agent, mins = de.evolve(precision=precision, print_steps=False)
                results += [mins]
                de.__init__(crossover=crossover, mutate=mutate, size=size, eval_limit=eval_limit,
                            benchmark=benchmark)
                print(benchmark.name, j+1, agent.value)
            self.csv(results, "DE_Results/de_output_"+benchmark.name+".csv")

    def GA(self, evaluations=100000, crossover=0.8, mutate=0.2, size=100, benchmarks=(ComparativeBenchmarks.f1()),
           precision=None):
        """
        Run GA algorithm on list of benchmarks. 30 executions per benchmark.

        :param evaluations:
        :param crossover:
        :param mutate:
        :param size:
        :param benchmarks:
        :param precision:
        """
        for i in range(len(benchmarks)):
            benchmark = benchmarks[i]
            ga = GeneticAlgorithm(crossovers=crossover, mutations=mutate, size=size, eval_limit=evaluations,
                                  benchmark=benchmark)
            results = []
            for j in range(30):
                ind, mins = ga.evolve(precision=precision, print_steps=False)
                results += [mins]
                ga.__init__(crossovers=crossover, mutations=mutate, size=size, eval_limit=evaluations,
                            benchmark=benchmark)
                print(i+13, j+1, ind.value)
            self.csv(results, "GA_Results/ga_output_"+str(i+13)+".csv")

    def PSO(self, evaluations=100000, inertia_weight=0.2, cognitive_constant=1.8, social_constant=1.8,
            size=100, benchmarks=(ComparativeBenchmarks.f1()), precision=None):
        """
        Run PSO algorithm on list of benchmarks. 30 executions per benchmark.

        :param evaluations:
        :param inertia_weight:
        :param cognitive_constant:
        :param social_constant:
        :param size:
        :param benchmarks:
        :param precision:
        """
        for i in range(len(benchmarks)):
            benchmark = benchmarks[i]
            pso = ParticleSwarmOptimisation(inertia_weight=inertia_weight, cognitive_constant=cognitive_constant,
                                            social_constant=social_constant, size=size, eval_limit=evaluations,
                                            benchmark=benchmark)
            results = []
            for j in range(30):
                particle, mins = pso.swarm(precision=precision, print_steps=False)
                results += [mins]
                pso.__init__(inertia_weight=inertia_weight,
                             cognitive_constant=cognitive_constant, social_constant=social_constant, size=size,
                             eval_limit=evaluations, benchmark=benchmark)
                print(benchmark.name, j+1, particle.value)
            self.csv(results, "PSO_Results/pso_output_"+benchmark.name+".csv")

    def SA(self, evaluations=100000, max_temp=None, cool_rate=0.5, benchmarks=(ComparativeBenchmarks.f1()),
           precision=None):
        """
        Run SA algorithm on list of benchmarks. 30 executions per benchmark.

        :param evaluations:
        :param max_temp:
        :param cool_rate:
        :param benchmarks:
        :param precision:
        """
        for i in range(len(benchmarks)):
            benchmark = benchmarks[i]
            sa = SimulatedAnnealing(max_temp=max_temp, cool_rate=cool_rate, eval_limit=evaluations, benchmark=benchmark)
            results = []
            for j in range(30):
                state, mins = sa.annealing(precision=precision, print_steps=False)
                results += [mins]
                sa.__init__(max_temp=max_temp, cool_rate=cool_rate, eval_limit=evaluations, benchmark=benchmark)
                print(benchmark.name, j+1, state.value)
            self.csv(results, "SA_Results/sa_output_"+benchmark.name+".csv")


if __name__ == "__main__":
    ca = CompareAlgorithms()
    benchmarks = ComparativeBenchmarks.benchmarks()
    ca.ABC(benchmarks=benchmarks)
