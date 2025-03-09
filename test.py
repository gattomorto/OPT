import unittest
import Frank_Wolfe as fw
import torch
import torch.nn as nn
import random


class T(unittest.TestCase):
    def test_is_on_boundary(self):
        tol = 0.01
        C = fw.Box(  torch.tensor([ [-1,4] , [0,1] , [-4,6] ], dtype=torch.float32),tol =tol )
        x1 = torch.tensor([0, 0.5, -1], dtype=torch.float32)
        x2 = torch.tensor([-1, 0.5, 5], dtype=torch.float32)
        x3 = torch.tensor([-1, 1, -4], dtype=torch.float32)
        x4 = torch.tensor([-1-tol/2, 1, -4], dtype=torch.float32)
        x5 = torch.tensor([-1-tol/2, 1+tol/2, -4-tol/2], dtype=torch.float32)
        x6 = torch.tensor([-1-tol*2, 1+tol/2, -4-tol/2], dtype=torch.float32)
        x7 = torch.tensor([-1, 0.5, -4-tol/2], dtype=torch.float32)
        x8 = torch.tensor([-1, 0.5, -4-tol-tol/2], dtype=torch.float32)
        self.assertFalse(C.is_on_boundary(x1), "x1 is not on the boundary of C")
        self.assertTrue(C.is_on_boundary(x2), "x2 is on the boundary of C")
        self.assertTrue(C.is_on_boundary(x3), "x3 is on the boundary of C")
        self.assertTrue(C.is_on_boundary(x4), "x4 is on the boundary of C")
        self.assertTrue(C.is_on_boundary(x5), "x5 is on the boundary of C")
        self.assertFalse(C.is_on_boundary(x6), "x6 not on the boundary of C")
        self.assertTrue(C.is_on_boundary(x7), "x7 is on the boundary of C")
        self.assertFalse(C.is_on_boundary(x8), "x8 is not on the boundary of C")


        C = fw.L2_Ball(torch.tensor([1., -1.]), 1, tol)
        self.assertTrue(C.is_on_boundary(torch.tensor([1, 0])))
        self.assertTrue(C.is_on_boundary(torch.tensor([0, -1])))
        self.assertTrue(C.is_on_boundary(torch.tensor([-tol/2, -1])))
        self.assertFalse(C.is_on_boundary(torch.tensor([-tol*2, -1])))


        C = fw.Unit_Simplex(tol = 0.01)
        self.assertTrue(C.is_on_boundary(torch.tensor([1, 0, 0])))
        self.assertFalse(C.is_on_boundary(torch.tensor([0.5, 0.1, 0.4])))
        self.assertFalse(C.is_on_boundary(torch.tensor([-0.1, 1.1, 0.0])))
        self.assertFalse(C.is_on_boundary(torch.tensor([0, 1.1, 0.2])))
        self.assertTrue(C.is_on_boundary(torch.tensor([0, 0.8, 0.2])))
        self.assertTrue(C.is_on_boundary(torch.tensor([tol/2, 0.8, 0.2])))
        self.assertTrue(C.is_on_boundary(torch.tensor([tol/4, 0.8 + tol/2, 0.2])))
        self.assertFalse( C.is_on_boundary(torch.tensor([tol/2, 0.8+tol*2, 0.2])))

    def test_is_in_corner(self):
        tol = 0.1
        C = fw.LInf_Ball(torch.tensor([0, 0, 0], dtype=torch.float32), radius=1, tol=tol)

        x = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.assertFalse(C.is_in_corner(x), "x is not in the corner")

        x = torch.tensor([1, 1, 1], dtype=torch.float32)
        self.assertTrue(C.is_in_corner(x), "x is in the corner")

        x = torch.tensor([1, 1, 0], dtype=torch.float32)
        self.assertFalse(C.is_in_corner(x), "x is not in the corner")

        x = torch.tensor([2, 1, 0], dtype=torch.float32)
        self.assertFalse(C.is_in_corner(x), "x is not in the corner")

        x = torch.tensor([-1, 1, 1], dtype=torch.float32)
        self.assertTrue(C.is_in_corner(x), "x is in the corner")

        x = torch.tensor([1, -1 - tol/2, 1], dtype=torch.float32)
        self.assertTrue(C.is_in_corner(x), "x is in the corner")

        x = torch.tensor([1 + tol/2, 1 + tol/2, 1 + tol/2], dtype=torch.float32)
        self.assertTrue(C.is_in_corner(x), "x is in the corner")

        x = torch.tensor([1 + tol/2, 1 + 3/2*tol, 1 + tol/2], dtype=torch.float32)
        self.assertFalse(C.is_in_corner(x), "x is not in the corner")

    def test_project_to_boundary(self):
        tol = 1e-5
        C = fw.Box(  torch.tensor([[-3,3],[-1,1]], dtype=torch.float32) )
        x = torch.tensor([1,1/2], dtype=torch.float32)
        d = torch.tensor([1,2], dtype=torch.float32)
        p = C.project_onto_the_boundary(x,d,"vec")
        alpha = C.project_onto_the_boundary(x,d,"alpha")
        self.assertTrue( torch.abs(alpha-1/4) < tol,"alpha = 1/4")
        self.assertTrue( torch.max(torch.abs(p - (x+1/4*d))) < tol, "projection = [1.25, 1]" )

        C = fw.Box(  torch.tensor([[-7,1],[-3,-1]], dtype=torch.float32) )
        x = torch.tensor([-3,-2], dtype=torch.float32)
        d = torch.tensor([-1,-1/2], dtype=torch.float32)
        p = C.project_onto_the_boundary(x,d,"vec")
        alpha = C.project_onto_the_boundary(x,d,"alpha")
        self.assertTrue( torch.abs(alpha-2.0) < tol,"alpha = 2")
        self.assertTrue( torch.max(torch.abs(p - (x+2*d))) < tol, "projection = [-5, -3]" )

        C = fw.Box(  torch.tensor([[-7,1],[-3,-1]], dtype=torch.float32) )
        x = torch.tensor([0,-2], dtype=torch.float32)
        d = torch.tensor([-1,-1], dtype=torch.float32)
        p = C.project_onto_the_boundary(x,d,"vec")
        alpha = C.project_onto_the_boundary(x,d,"alpha")
        self.assertTrue( torch.abs(alpha-1.0) < tol,"alpha = 1")
        self.assertTrue( torch.max(torch.abs(p - (x+1*d))) < tol, "projection = [-1, -3]" )

        # x already on boundary
        C = fw.Box(  torch.tensor([[-7,1],[-3,-1]], dtype=torch.float32) )
        x = torch.tensor([0,-1], dtype=torch.float32)
        d = torch.tensor([-1,-1], dtype=torch.float32)
        p = C.project_onto_the_boundary(x,d,"vec")
        alpha = C.project_onto_the_boundary(x,d,"alpha")
        self.assertTrue( torch.abs(alpha-0.0) < tol,"alpha = 0")
        self.assertTrue( torch.max(torch.abs(p - (x+0*d))) < tol, "projection = [0, -1]" )

    def test_violation(self):
        tol = 0.1
        C = fw.Box(torch.tensor([[-1, 4], [0, 1], [-4, 6]]), tol=tol)

        x0 = torch.tensor([0, 1, -4])
        v0 = C.violation(x0)
        self.assertAlmostEqual(v0,0.,7)

        x1 = torch.tensor([0, 0.5, 0])
        v1 = C.violation(x1)
        self.assertAlmostEqual(v1,0.,7)

        x2 = torch.tensor([0, 1, 0])
        v2 = C.violation(x2)
        self.assertAlmostEqual(v2,0.,7)

        x4 = torch.tensor([0, 1 + tol/2, 0])
        v4 = C.violation(x4)
        self.assertAlmostEqual(v4,0.,7)

        x5 = torch.tensor([0, 1 + tol*2, 0])
        v5 = C.violation(x5)
        self.assertAlmostEqual(v5,tol,7)

        #L2-Ball
        C = fw.L2_Ball(torch.tensor([0., -1]),1,tol)

        v1 = C.violation(torch.tensor([0, 0]))
        self.assertAlmostEqual(v1,0,7)

        v2 = C.violation(torch.tensor([0, -1]))
        self.assertAlmostEqual(v2,0,7)

        v3 = C.violation(torch.tensor([0, 1]))
        self.assertAlmostEqual(v3,1-tol,7)

        v4 = C.violation(torch.tensor([1, 0]))
        self.assertAlmostEqual(v4,(torch.sqrt(torch.tensor(2))-1.).item()-tol,7)

        v5 = C.violation(torch.tensor([0, -tol/2]))
        self.assertAlmostEqual(v5,0,7)

        v6 = C.violation(torch.tensor([0, tol*2]))
        self.assertAlmostEqual(v6,tol,7)

        pass


    def test_contains(self):
        tol = 0.1
        #Box
        C = fw.Box(torch.tensor([[-1, 4], [0, 1], [-4, 6]]), tol=tol)
        self.assertTrue(C.__contains__(torch.tensor([0, 0.5, 0])))
        self.assertTrue(C.__contains__(torch.tensor([0, 1, -4])))
        self.assertTrue(C.__contains__(torch.tensor([0, 1+tol/2, -4-tol/2])))
        self.assertFalse(C.__contains__(torch.tensor([0, 1+tol/2, -4-tol*2])))

        #L2-Ball
        C = fw.L2_Ball(torch.tensor([0., -1]), 1, tol)
        self.assertTrue(C.__contains__(torch.tensor([0, -1])))
        self.assertTrue(C.__contains__(torch.tensor([0.5, -1])))
        self.assertTrue(C.__contains__(torch.tensor([0, 0])))
        self.assertTrue(C.__contains__(torch.tensor([0, tol/2])))
        self.assertTrue(C.__contains__(torch.tensor([0-tol/2, tol/2])))
        self.assertTrue(C.__contains__(torch.tensor([0+tol/2, tol/2])))
        self.assertFalse(C.__contains__(torch.tensor([0, tol*2])))

        # Unit Symplex
        C = fw.Unit_Simplex(tol = tol)
        self.assertTrue(C.__contains__(torch.tensor([0, 1])))
        self.assertTrue(C.__contains__(torch.tensor([1, 0])))
        self.assertTrue(C.__contains__(torch.tensor([0.5, 0.5])))
        self.assertTrue(C.__contains__(torch.tensor([0.1, 0.9])))
        self.assertFalse(C.__contains__(torch.tensor([-0.1, 1.1])))
        self.assertTrue(C.__contains__(torch.tensor([0.1, 0.9+tol/2])))
        self.assertTrue(C.__contains__(torch.tensor([0.1+tol/4, 0.9+tol/2])))
        self.assertFalse(C.__contains__(torch.tensor([0.1+tol*3/2, 0.9+tol/2])))

    def test_intetsection(self):
        C1 = fw.Box(torch.tensor([[-1, 4], [0, 1], [-4, 6]]), tol=0.1)
        C2 = fw.LInf_Ball(torch.tensor([0, 0, 0]), radius=1, tol=0.2)
        C = C1*C2
        self.assertTrue(C.tol==C2.tol)

class T2(unittest.TestCase):
    class ConvexFunction1(torch.nn.Module):
        def forward(self, xx):
            x = xx[0]
            y = xx[1]
            # z = xx[2]

            # return torch.square(x) + torch.square(y) + torch.square(z)
            return 10 * torch.square(x) + torch.square(y)
            # return (x-5)**2 + (y-4)**2 + (z-3)**2
            # return torch.exp(x*y*z)
    class ConvexFunction2(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.quadratic_weights = nn.Parameter(torch.eye(input_dim))
            self.linear_weights = nn.Parameter(torch.zeros(input_dim))
            self.bias = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(0)
            quadratic_term = torch.sum(x @ self.quadratic_weights * x, dim=1)
            linear_term = torch.sum(x * self.linear_weights, dim=1) + self.bias
            nonsmooth_term = self.alpha * torch.max(torch.abs(x), dim=1).values
            smooth_term = self.beta * torch.log1p(torch.linalg.norm(x, dim=1) ** 2)
            return quadratic_term + linear_term + nonsmooth_term + smooth_term

    def test_safw(self):
        f = self.ConvexFunction1()
        C = fw.LInf_Ball(torch.tensor([0, 2],dtype=torch.float64), 1)
        x_0 = torch.tensor([0.5, 2.5])
        #_, fsas = fw.SAFWf(f, x_0, C, 1000, 0, grad_tol=0, verbose=False)
        _, fsas = fw.UFW("asfw",f, x_0, C, 1000, 0, grad_tol=0, verbose=False)
        l1 = len(fsas)
        self.assertTrue((l1 == 16))

        input_dim = 30
        x_0 = torch.tensor([0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341, 0.4901, 0.8964, 0.4556,
                            0.6323, 0.3489, 0.4017, 0.0223, 0.1689, 0.2939, 0.5185, 0.6977, 0.8000,
                            0.1610, 0.2823, 0.6816, 0.9152, 0.3971, 0.8742, 0.4194, 0.5529, 0.9527,
                            0.0362, 0.1852, 0.3734]) * 10
        C = fw.LInf_Ball(x_0, radius=1)
        f = self.ConvexFunction2(input_dim)
        verbose = False
        max_iter = 1000
        _, fsas = fw.UFW("asfw",f, x_0, C, max_iter, 0, grad_tol=0, verbose=verbose)
        l1 = len(fsas)
        self.assertTrue( (l1 == 37))

    def test_pwfw(self):
        torch.manual_seed(0)
        input_dim = 30
        x_0 = torch.tensor([0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341, 0.4901, 0.8964, 0.4556,
                            0.6323, 0.3489, 0.4017, 0.0223, 0.1689, 0.2939, 0.5185, 0.6977, 0.8000,
                            0.1610, 0.2823, 0.6816, 0.9152, 0.3971, 0.8742, 0.4194, 0.5529, 0.9527,
                            0.0362, 0.1852, 0.3734]) * 10
        C = fw.LInf_Ball(x_0, radius=1)
        f = self.ConvexFunction2(input_dim)
        verbose = False
        max_iter = 1000
        #_, fspw = fw.PWFWf(f, x_0, C, max_iter, 0, grad_tol=0, verbose=verbose)
        _, fspw = fw.UFW("pwfw",f, x_0, C, max_iter, 0, grad_tol=0, verbose=verbose)
        l1 = len(fspw)
        self.assertTrue( (l1 == 29))

    def test_fw(self):
        torch.manual_seed(0)
        input_dim = 30
        x_0 = torch.tensor([0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341, 0.4901, 0.8964, 0.4556,
                            0.6323, 0.3489, 0.4017, 0.0223, 0.1689, 0.2939, 0.5185, 0.6977, 0.8000,
                            0.1610, 0.2823, 0.6816, 0.9152, 0.3971, 0.8742, 0.4194, 0.5529, 0.9527,
                            0.0362, 0.1852, 0.3734]) * 10
        C = fw.LInf_Ball(x_0, radius=1)
        f = self.ConvexFunction2(input_dim)
        verbose = False
        max_iter = 1000
        _, fsfw = fw.UFW("fw",f, x_0, C, max_iter, 0, grad_tol=0, verbose=verbose)
        l2 = len(fsfw)
        self.assertTrue((l2 == 216))


if __name__ == '__main__':
    unittest.main()


import torch
import torch.nn as nn
import cvxpy as cp
from scipy.optimize import minimize
import numpy as np
import Frank_Wolfe

'''
UNIT SIMPLEX

Optimal value: -0.1102842527162104
Optimal x: 0.0
Optimal y: 0.7913682278622345
Optimal z: 0.2086317721377656
return np.square(x)*y - y*np.sin(z) + np.exp(z)*np.square(z)

Optimal value: 0.16285657058387792
Optimal x: 0.0
Optimal y: 0.7050506927343316
Optimal z: 0.29494930726567153
return 1/3*np.cos(np.square(x)-x*y*z) - y*np.sin(z) + np.tan(np.exp(z)*np.square(z)*z)

Optimal value: 0.18010076862271343
Optimal x: 0.9999999999999997
Optimal y: 2.498001805406602e-16
Optimal z: 4.163336342344337e-17
return 1/3*np.cos(np.square(x)-x*y*z) - y*np.sin(z)**2 + np.tan(np.exp(z)*np.square(z)*z)/x

Optimal value: -1.000000000385349
Optimal x: 0.0
Optimal y: 0.0
Optimal z: 1.0000000001284497
return np.cos(x)*y - z**3 + x*z

Optimal value: -34.21107192536885
Optimal x: 1.3877787807814457e-15
Optimal y: 0.49999999999999967
Optimal z: 0.4999999999999989
nota che esistono infinite soluzioni (x=0, y=a, z = 1-a)

Optimal value: 0.4804530139182096
Optimal x: 0.4999999999999969
Optimal y: 0.49999999999999717
Optimal z: 5.88418203051333e-15
return np.log(x)*np.log(y)

aumentando la tolleranza dà questo, ma dà errore
Optimal value: -inf
Optimal x: 0.43168967089493504
Optimal y: 0.5683103291050656
Optimal z: 0.0

la soluzione ideale è (a, 1-a, 0)
Optimal value: 1.0
Optimal x: 0.4999999999999999
Optimal y: 0.5
Optimal z: 5.551115123125783e-17
return np.exp(z)

la soluzione ideale è (0,0,1)
Optimal value: 1.0000000000000002
Optimal x: 2.220446049250313e-16
Optimal y: 0.0
Optimal z: 0.9999999999999998
return np.exp(x)*np.exp(y)

ogni punto del simplex è minima
Optimal value: 2.718281828459045
Optimal x: 0.3333333333333333
Optimal y: 0.3333333333333333
Optimal z: 0.3333333333333333
return np.exp(x)*np.exp(y)*np.exp(z)

la soluzioni ideali hanno almeno uno 0
Optimal value: 1.0
Optimal x: 0.0
Optimal y: 0.38409173296247134
Optimal z: 0.6159082670375287
return np.exp(x*y*z)

# qui il minimo è all'interno di C
Optimal value: 0.0
Optimal x: 0.3333333333333333
Optimal y: 0.3333333333333333
Optimal z: 0.3333333333333333
return np.square(x-1/3) + np.square(y-1/3) + np.square(z-1/3)

def objective(vars):
    x, y, z = vars

    return np.square(x-1/3) + np.square(y-1/3) + np.square(z-1/3)

# Constraint: x + y + z = 1
def equality_constraint(vars):
    return np.sum(vars) - 1

# Bounds: x, y, z >= 0
bounds = [(0, None), (0, None), (0, None)]

# Initial guess: start with equal division of simplex
initial_guess = [1/6, 2/6, 3/6]

# Define constraints for scipy
constraints = {'type': 'eq', 'fun': equality_constraint}

# Solve the problem
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, tol=1e-40)

# Output the results
print("Optimal value:", result.fun)
print("Optimal x:", result.x[0])
print("Optimal y:", result.x[1])
print("Optimal z:", result.x[2])
'''

'''
Linf Ball

return np.square(x - 1 / 3) + np.square(y - 1 / 3) + np.square(z - 1 / 3)
Optimal values (x, y, z): [0.33333333 0.33333333 0.33333333]
Minimum value of f(x, y, z): 7.497813317466739e-17

return np.square(x) * y - y * np.sin(z) + np.exp(z) * np.square(z)
soluzioni: [0., 1, 0.35], [0,0,0], [1,-1,-1], [-1. -1. -1.]
se parti da [0,0,0]
Optimal values (x, y, z): [0. 0. 0.]
Minimum value of f(x, y, z): 0.0
--------------------------------------------
se parti da [-1,1,0]
Optimal values (x, y, z): [6.24306214e-08 1.00000000e+00 3.05021898e-01]
Minimum value of f(x, y, z): -0.1740931354687092
--------------------------------------------
se parti da [-1,1,-1] ma FW mi trova (0,0,0) come ottimo
Optimal values (x, y, z): [ 1. -1. -1.]
Minimum value of f(x, y, z): -1.4735915436364542
--------------------------------------------
se parti da [ 1. -1. -1.]
Optimal values (x, y, z): [ 1. -1. -1.]
Minimum value of f(x, y, z): -1.4735915436364542
--------------------------------------------
se parti da -1 -1 -1
Optimal values (x, y, z): [-1. -1. -1.]
Minimum value of f(x, y, z): -1.4735915436364542
--------------------------------------------
se parti da -1 1 -1, ma FW trova 0 0 0
Optimal values (x, y, z): [ 1. -1. -1.]
Minimum value of f(x, y, z): -1.4735915436364542
--------------------------------------------
se parti da -1 0 -1
Optimal values (x, y, z): [-1. -1. -1.]
Minimum value of f(x, y, z): -1.473591543636454
--------------------------------------------
partendo da 1 1 1, FW trova tensor([4.8340e-05, 9.9703e-01, 3.0443e-01], requires_grad=True)
Optimal values (x, y, z): [-1. -1. -1.]
Minimum value of f(x, y, z): -1.4735915436364542
--------------------------------------------


soluzioni: [0.2  1. 0.41], [1.0, 1.0, 0.38], [0. 1. 0.41], [0,0,0], [1,0,0], [-1,-1,-1], [0,-1,1] et al.
--------------------------------------------
se pari da [0, 1, 0.3]
Optimal values (x, y, z): [0. 1. 0.4172749]
Minimum value of f(x, y, z): 0.0387893834667474
return 1/3*np.cos(np.square(x)-x*y*z) - y*np.sin(z) + np.tan(np.exp(z)*np.square(z)*z)
--------------------------------------------
se parti da  0 0 0
Optimal values (x, y, z): [0. 0. 0.]
Minimum value of f(x, y, z): 0.3333333333333333
--------------------------------------------
se parti da 1 0 0
Optimal values (x, y, z): [1. 0. 0.]
Minimum value of f(x, y, z): 0.1801007686227132


unica soluzione
x_0:  [0, 0, -1]
Optimal values (x, y, z): [-1. -1. -1.]
Minimum value of f(x, y, z): 0.04978706836786395
return np.exp(x) * np.exp(y) * np.exp(z)

# 0,0,0 ce un ottimo locale, un numero dispari di -1 è un ottimo globale
return np.exp(x * y * z)

# Example objective function: Replace this with your function
def objective(vars):
    x, y, z = vars
    return np.exp(x * y * z)


# Bounds for the infinity norm constraint ||(x, y, z)||_inf <= 1
bounds = [(-1, 1), (-1, 1), (-1, 1)]

# Initial guess
initial_guess = [1/3, 1/3, 1/3]  # A point within the bounds

# Run the optimization
result = minimize(objective, initial_guess, bounds=bounds)

# Output the results
if result.success:
    print("x_0: ", initial_guess)
    print("Optimal values (x, y, z):", result.x)
    print("Minimum value of f(x, y, z):", result.fun)
else:
    print("Optimization failed:", result.message)





'''

'''
BOX bounds = [(-1, 2), (0, 1), (2, 5)]

unica soluzione di return x**2 + y**2 + z**2 è [0 0 2]
unica soluzione di  np.exp(x) + np.exp(y) + np.exp(z) è [-1, 0, 2]
[0, 1, 2], [-1, 0, 2] di np.square(x) * y - y * np.sin(z) + np.exp(z) * np.square(z)

def objective(vars):
    x, y, z = vars
    # Example: A simple quadratic function
    return np.square(x) * y - y * np.sin(z) + np.exp(z) * np.square(z)
bounds = [(-1, 2), (0, 1), (2, 5)]
initial_guess = np.array((0,0.5,3))
result = minimize(objective, initial_guess, bounds=bounds, method='TNC')
# Output the results
if result.success:
    print("Optimal values (x, y, z):", result.x)
    print("Minimum value of the objective function:", result.fun)
else:
    print("Optimization failed:", result.message)
'''

'''
L2_Ball(torch.tensor([2,-3,5], dtype=torch.float32),3)

f(x*):  10.01351598218614
unica sol: 1.0267, -1.5401,  2.5668 di x**2 + y**2 + z**2

f(x*):  1.6652149721507388e-53
[ 4.0512, -4.7496,  6.3158] unica sol di return np.exp(x*y*z) , ricorda di mettere tolleranze piccole perche credo che la funzoine è tanto piatta

def objective(vars):
    x, y, z = vars
    # Example: A simple quadratic function
    return np.exp(x*y*z)

def l2_ball_constraint(vars, center, radius):
    x, y, z = vars
    return radius**2 - np.sum((np.array([x, y, z]) - center)**2)

center = np.array([2, -3, 5])
radius = 3

# Nonlinear constraint for the L2 ball
constraint = {
    'type': 'ineq',  # Inequality constraint (>= 0)
    'fun': lambda vars: l2_ball_constraint(vars, center, radius)
}

# Initial guess
initial_guess = [2, -1, 5]  # A point inside the L2 ball

# Run the optimization
opt = {'disp': True,'maxiter':1000000000}
result = minimize(objective, initial_guess, constraints=[constraint], tol = 1e-30,options=opt)


# Output the results
if result.success:
    print("Optimal values (x, y, z):", result.x)
    print("Minimum value of the objective function:", result.fun)
else:
    print("Optimization failed:", result.message)
'''

