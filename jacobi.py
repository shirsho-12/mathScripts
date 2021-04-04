import numpy as np
from sympy import sin, cos, exp, Matrix, pprint, simplify, diff, symbols, S, limit, oo, atanh, Function, ln, log, pi, sqrt
from sympy.abc import phi, theta
import sympy
from sympy.calculus.util import continuous_domain


def calculate_jacobians(equation_matrix, num_vars, denom_vars):
    mat_1 = equation_matrix.jacobian(num_vars)
    mat_2 = equation_matrix.jacobian(denom_vars)

    det_1 = simplify(mat_1.det())
    pprint(mat_1)
    print(det_1)

    print('\n\n')
    pprint(mat_2)
    det_2 = simplify(mat_2.det())
    print(det_2)
    print('\nAnswer: ')
    ans = simplify(-det_1/det_2)
    R, x, y, z = symbols('R x y z')
    u, v = symbols('u v')
    # evaluated = ans.subs(*zip([x, y, z, u, v], [1, 1, 1, 1, 1]))
    args = [x, y, z, u, v]
    vals = [1, 1, 1, 1, 1]
    evaluated = ans.subs({"x":1, "y":1, "z":1, "u":1, "v":1})

    print(evaluated)
    pprint(ans)


def calculate_partial_derivatives(f, var):
    derivative = diff(f, var)
    return derivative


def get_domain(f, var, lim):
    pprint(limit(f, var, lim))
    print(continuous_domain(f, var, S.Reals))


def calculate_implicit(f, numerator_var, denom_var):
    num = calculate_partial_derivatives(f, numerator_var)
    denom = calculate_partial_derivatives(f, denom_var)
    pprint(simplify(-num/denom))


R, x, y, z = symbols('R x y z')
u, v = symbols('u v')
get_domain((sin(pi * x) + sin(3*pi/2))/log(2 - 2*x), x, 0.5)
# eqns = Matrix([x*y**2 + z*u + v**2, z*x**2 + 2*y - u*v, x*u + y*v - x*y*z])
# denominator = Matrix([x, y, z])
# numerator = Matrix([x, u, z])
# calculate_jacobians(eqns, numerator, denominator)
# print('\n')
#
s, t = symbols('s t')
# # pprint(simplify(calculate_partial_derivatives(x**2 - 2*y**2*s**2*t - 2*s*t**2 - 1, y)))
#
# # get_domain(sin(3*x)/x**3 + 4.5 - 6/(2*x**2), x, 0)
a = symbols('a')
# get_domain((ln(1-x) + x + 0.5*x**2)/x**3, x, 0)
# pprint(simplify(calculate_partial_derivatives(atanh(x), x)))

# f = symbols('f', cls=Function)
# z = exp(2*u)*v**3
# y = u * v - v**2
# x = u**3 + v ** 3
# R = f(x, y, z)
# pprint((simplify(diff(z, u) * diff(y, u) + diff(z, v) * diff(y, v))).evalf(subs={u: 1, v: 1}))
