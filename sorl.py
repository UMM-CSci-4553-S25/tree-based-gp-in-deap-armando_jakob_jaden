import operator
import math
import random

import numpy

from deap import algorithms, base, creator, tools, gp



# Define new functions

def less_than(a: int, b: int) -> bool:
    return a < b

def greater_or_equal(a: int, b: int) -> bool:
    return a >= b

def greater_than(a: int, b: int) -> bool:
    return a > b


pset = gp.PrimitiveSet("MAIN", [int], str)