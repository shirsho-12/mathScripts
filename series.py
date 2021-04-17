import numpy
from sympy import Symbol, pprint, simplify
import sympy as sp


def get_series(var, expr, num_terms=10):
    series = sp.series(expr, var, n=num_terms)
    pprint(simplify(series))


x = Symbol("x")
expr = sp.ln(1 - 8*x**2)
# expr = sp.cos(x)
# expr = sp.atan(x**3)
# expr = sp.ln(sp.sec(x))
get_series(x, expr)
