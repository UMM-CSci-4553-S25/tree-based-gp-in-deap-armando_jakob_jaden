import operator
import math
import random

import numpy

from deap import algorithms, base, creator, tools, gp


# Define new functions
def SmallOrLarge(x):
    if x >= 1000 & x < 2000:
        return x
    elif x >= 2000:
        return 1
    else:
        return 0 # We can change the 0 and 1 to Small and Large to fit with the problem

def IntToString(x):
    if x == 0:
        return "Small"
    elif x == 1:
        return "Large"
    else:
        return "Mid" 
    
