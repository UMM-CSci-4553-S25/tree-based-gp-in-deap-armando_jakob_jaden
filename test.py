import operator
import random
import functools

import numpy

from deap import algorithms, base, creator, tools, gp

# ----------------------------
# Define Boolean Logic Functions
# ----------------------------

def less_than(a: int, b: int) -> bool:
    return a < b

def greater_or_equal(a: int, b: int) -> bool:
    return a >= b

def greater_than(a: int, b: int) -> bool:
    return a > b

def if_then_else(condition: bool, true_branch: bool, false_branch: bool) -> bool:
    return true_branch if condition else false_branch

# ----------------------------
# Primitive Set Definition
# ----------------------------

pset = gp.PrimitiveSetTyped("MAIN", [int, int], bool)
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="i")

# Logic primitives
pset.addPrimitive(less_than, [int, int], bool, name="lt")
pset.addPrimitive(greater_or_equal, [int, int], bool, name="ge")
pset.addPrimitive(greater_than, [int, int], bool, name="gt")
pset.addPrimitive(operator.eq, [int, int], bool, name="eq")
pset.addPrimitive(operator.ne, [int, int], bool, name="ne")
pset.addPrimitive(if_then_else, [bool, bool, bool], bool, name="if")

# Arithmetic primitives
pset.addPrimitive(operator.add, [int, int], int, name="add")
pset.addPrimitive(operator.sub, [int, int], int, name="sub")

# Terminals
pset.addTerminal(True, bool)
pset.addTerminal(False, bool)
pset.addEphemeralConstant("randInt", functools.partial(random.randint, 0, 10), int)

# ----------------------------
# Setup GP Structure
# ----------------------------

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# ----------------------------
# Fitness Evaluation
# ----------------------------

def evalLogic(individual, points):
    func = toolbox.compile(expr=individual)
    fitness = 0
    for x, i in points:
        try:
            result = func(x, i)
            if result == (x < i):
                fitness += 1
        except Exception:
            pass  # Penalize by not increasing fitness
    return fitness,

# Training/Test split
full_data = [(random.randint(0, 10), random.randint(0, 10)) for _ in range(60)]
training_data = full_data[:40]
test_data = full_data[40:]

toolbox.register("evaluate", evalLogic, points=training_data)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Limit tree height
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# ----------------------------
# Main Execution
# ----------------------------

def main() -> tuple[list, object, tools.HallOfFame]:
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=100, lambda_=200,
        cxpb=0.5, mutpb=0.1, ngen=100,
        stats=mstats, halloffame=hof, verbose=True
    )

    print("\nBest individual:")
    print(str(hof[0]))
    func = toolbox.compile(expr=hof[0])
    
    test_correct = sum(1 for x, i in test_data if func(x, i) == (x < i))
    print(f"Test set accuracy: {test_correct} / {len(test_data)}")

    return pop, log, hof

if __name__ == "__main__":
    main()
