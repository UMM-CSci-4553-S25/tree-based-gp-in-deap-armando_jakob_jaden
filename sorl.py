
import operator
import math
import random

import numpy

from functools import partial
from deap import algorithms, base, creator, tools, gp

# Define new functions

def greaterthan(a: int, b: int) -> bool:
    """Check if a is greater than n."""
    return a > b

def lessthan(a: int, b: int) -> bool:
    """Check if a is less than n."""
    return a < b

def equal(a: int, b: int) -> bool:
    """Check if a is equal to n."""
    return a == b

pset = gp.PrimitiveSet("MAIN", 2)

pset.addPrimitive(greaterthan, 2, name="greaterthan")
pset.addPrimitive(lessthan, 2, name="lessthan")

pset.addPrimitive(equal, 2, name="equal")
pset.renameArguments(ARG0="a", ARG1="b")


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    """Evaluate the symbolic regression model."""
    func = toolbox.compile(expr=individual)
    sqerrors = ((func(x, y) - x) ** 2 for x, y in points)
    return math.fsum(sqerrors) / len(points),

training_inputs = [(random.randint(-10000, 10000), 200 * random.random() - 100) for index in range(0, 40)]

print(training_inputs)

toolbox.register("evaluate", evalSymbReg, points=training_inputs)

toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("mate", gp.cxOnePoint)

toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)

    for winner in hof:
        print(str(winner))

    return pop, log, hof

if __name__ == "__main__":
    main()