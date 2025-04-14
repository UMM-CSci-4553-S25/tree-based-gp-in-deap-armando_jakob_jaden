import operator
import random
import numpy
import math

from functools import partial
from deap import algorithms, base, creator, tools, gp

# Define the comparison and label functions
def greater_than(x: int, y: int) -> bool:
    return x > y

def less_than(x: int, y: int) -> bool:
    return x < y

def equals(x: int, y: int) -> bool:
    return x == y

def label_large(x: int) -> str:
    # print("large")
    return "large"

def label_small(x: int) -> str:
    # print("small")
    return "small"

def if_then_else(condition: bool, out1, out2):
    return out1 if condition else out2

# Set up the GP primitive set with one argument
pset = gp.PrimitiveSet("MAIN", 1)
pset.renameArguments(ARG0="x")

# Add primitives and terminals
pset.addPrimitive(greater_than, 2, name="gt")
pset.addPrimitive(less_than, 2, name="lt")
pset.addPrimitive(equals, 2, name="eq")
pset.addPrimitive(label_large, 1)
pset.addPrimitive(label_small, 1)
pset.addPrimitive(if_then_else, 3)
pset.addTerminal(1000)
pset.addTerminal(2000)
pset.addTerminal("none")  # Fallback for silent case

# Define GP individual and fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# GP structure generation
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Evaluation function for classification behavior
def evalClassifier(individual):
    func = toolbox.compile(expr=individual)
    correct_behavior = 0

    test_inputs = [random.randint(-10000, 10000) for _ in range(20)]

    for x in test_inputs:
        try:
            output = func(x)
        except Exception:
            output = None

        if x >= 2000 and output == "large":
            correct_behavior += 1
        elif x < 1000 and output == "small":
            correct_behavior += 1
        elif 1000 <= x < 2000 and output == "none":
            correct_behavior += 1

    return 1.0 - (correct_behavior / 20),

toolbox.register("evaluate", evalClassifier)
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

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 500, stats=mstats,
                                   halloffame=hof, verbose=True)

    for winner in hof:
        print("Best individual:")
        print(str(winner))

        return pop, log, hof

if __name__ == "__main__":
    main()
