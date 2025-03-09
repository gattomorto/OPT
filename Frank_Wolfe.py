import sys
from abc import ABC, abstractmethod
import torch
from typing import final

torch.set_default_dtype(torch.float64)

class Feasible_Set(ABC):
    def __init__(self, tol = None):
        super().__init__()
        self.tol = tol if tol is not None else 1e-6

    @abstractmethod
    def __contains__(self, x):
        pass

    @abstractmethod
    def violation(self, x: torch.Tensor):
        pass

    @staticmethod
    def intersect(C1, C2):
        if not isinstance(C1, Box) or not isinstance(C2, Box):
            raise ValueError("Intersection is only supported between Box constraints.")

        return C1*C2

    @abstractmethod
    def is_on_boundary(self, x):
        pass

    @final
    def lm(self, x_k: torch.Tensor, grad_tol = None):
        grad_norm = torch.linalg.vector_norm(x_k.grad)
        grad_tol = grad_tol if grad_tol is not None else 1e-4
        if grad_norm < grad_tol:
            return x_k

        return self.lm_specific(x_k)

    @abstractmethod
    def lm_specific(self, x_k):
        pass

class Box(Feasible_Set):
    def __init__(self, bounds: torch.Tensor, tol = None):
        super().__init__( tol = tol)
        self.bounds = bounds

    def __contains__(self, x):
        '''lower_bounds = self.bounds[:, 0] - self.tol
        upper_bounds = self.bounds[:, 1] + self.tol
        return torch.all((x >= lower_bounds) & (x <= upper_bounds)).item()'''
        return self.violation(x) == 0

    def __mul__(self, other):
        new_lower_bounds = torch.maximum(self.bounds[:, 0], other.bounds[:, 0])
        new_upper_bounds = torch.minimum(self.bounds[:, 1], other.bounds[:, 1])

        if torch.any(new_lower_bounds > new_upper_bounds):
            raise ValueError("The intersection is the empty set")

        new_bounds = torch.stack([new_lower_bounds, new_upper_bounds], dim=1)
        return Box(new_bounds, tol=max(self.tol,other.tol))

    def violation(self, x):
        lower_violation = torch.max(self.bounds[:, 0] - x - self.tol, torch.tensor(0.0))
        upper_violation = torch.max(x - self.bounds[:, 1] - self.tol, torch.tensor(0.0))
        # violation in L-inf norm
        violation = torch.max(torch.max(lower_violation, upper_violation),torch.tensor(0.0))
        max_violation = torch.max(violation).item()
        return max_violation

    def is_on_boundary(self, x):
        touches_lower_boundary = torch.any((x >= self.bounds[:, 0] - self.tol) & (x <= self.bounds[:, 0] + self.tol))
        touches_upper_boundary = torch.any((x >= self.bounds[:, 1] - self.tol) & (x <= self.bounds[:, 1] + self.tol))

        # a point is on the boundary if it's within the box and touches any boundary
        return (x in self) and (touches_lower_boundary or touches_upper_boundary)

    def project_onto_the_boundary(self, x, d, return_type):
        '''
        given a point inside C, projects x in the direction d onto the boundary of C
        solves max alpha s.t. l_i <= x_i + alpha*d_i <= u_i
        '''
        if x not in self:
            raise ValueError("x is not in C")

        if self.is_on_boundary(x):
            if return_type == "vec":
                return x
            elif return_type == "alpha":
                return torch.tensor(0)

        lower_bounds = self.bounds[:, 0] - self.tol
        upper_bounds = self.bounds[:, 1] + self.tol
        alpha_max = float('inf')
        for i in range(len(x)):
            if d[i] > 0:
                alpha_i = (upper_bounds[i] - x[i]) / d[i]
            elif d[i] < 0:
                alpha_i = (lower_bounds[i] - x[i]) / d[i]
            else:
                alpha_i = float('inf')

            alpha_max = min(alpha_max, alpha_i)

        if return_type == "vec":
            return x + alpha_max*d
        elif return_type == "alpha":
            return alpha_max

    def is_in_corner(self, x):
        if x not in self:
            return False

        touches_lower_boundary = (x >= self.bounds[:, 0] - self.tol) & (x <= self.bounds[:, 0] + self.tol)
        touches_upper_boundary = (x >= self.bounds[:, 1] - self.tol) & (x <= self.bounds[:, 1] + self.tol)
        is_corner = torch.all(touches_lower_boundary | touches_upper_boundary)
        return is_corner.item()

    def lm_specific(self, x_k: torch.Tensor):
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        x_star = torch.where(x_k.grad > 0, lower_bounds, upper_bounds)
        return x_star

class LInf_Ball(Box):
    def __init__(self, center, radius, tol = None):
        self.center = center
        self.radius = radius

        lower_bounds = center - radius
        upper_bounds = center + radius
        bounds = torch.stack([lower_bounds, upper_bounds], dim=1)

        super().__init__(bounds,tol = tol)

class L2_Ball(Feasible_Set):
    def __init__(self, center: torch.Tensor, radius, tol=None):
        super().__init__(tol=tol)

        self.center = center
        self.radius = radius

    def __contains__(self, x: torch.Tensor):
        '''distance = torch.linalg.vector_norm(x - self.center)
        print('distance: ',distance)
        return distance <= self.radius + self.tol'''
        return self.violation(x) == 0

    def violation(self, x: torch.Tensor):
        d = torch.linalg.vector_norm(x-self.center) - self.tol
        v = torch.max(d-self.radius,torch.tensor(0.))
        return v.item()

    def is_on_boundary(self, x):
        d = torch.linalg.vector_norm(x-self.center)
        v = d-self.radius
        return -self.tol <= v and v <= self.tol

    def lm_specific(self, x_k: torch.Tensor):
        x_star = - x_k.grad/torch.linalg.vector_norm(x_k.grad)*self.radius + self.center
        if x_star not in self:
            sys.exit('x is outside C')
        return x_star

class Unit_Simplex(Feasible_Set):
    def __init__(self, tol = None):
        super().__init__(tol = tol)

    def __contains__(self, x):
        non_negative = torch.all(x >= 0)
        sum_within_tol = abs(torch.sum(x).item() - 1) <= self.tol
        return non_negative.item() and sum_within_tol

    def violation(self, x: torch.Tensor):
        pass

    def is_on_boundary(self, x):
        contains_zero = ((x >= -self.tol) & (x <= self.tol)).any()
        belongs_to_simplex = x in self
        return contains_zero and belongs_to_simplex

    def lm_specific(self, x_k: torch.Tensor):
        i_k = torch.argmin(x_k.grad)
        x_star = torch.zeros_like(x_k)
        x_star[i_k] = 1
        return x_star

class ActiveSet():
    def __init__(self,x_0,method):
        self.data = dict()
        self.method = method
        self[x_0]=1

    def hash_check(self,x):
        if not hasattr(x, 'hash'):
            x.hash = x.numpy().tobytes()

    def __contains__(self, x):
        self.hash_check(x)
        return x.hash in self.data

    def __getitem__(self, x):
        self.hash_check(x)
        # Return only the weight
        return self.data[x.hash][1]

    def __setitem__(self, x, w):
        self.hash_check(x)
        # Store the atom and weight as a tuple
        self.data[x.hash] = (x, w)

    def __delitem__(self, x):
        self.hash_check(x)
        del self.data[x.hash]

    def clear(self):
        self.data.clear()

    def atoms(self):# keys
        return [item[0] for item in self.data.values()]

    def __iter__(self):
        return iter(self.atoms())

    def __len__(self):
        return len(self.data)

    def items(self):
        # return tuples of (atom, weight)
        return [(item[0], item[1]) for item in self.data.values()]

    def current_iterate(self):
        # compute the sum of x * w for all (atom, weight) pairs
        return sum(x * w for x, w in self.items())

    def weights(self):
        return [item[1] for item in self.data.values()]

    def max_dot(self, x_k):
        max_dot = float('-inf')
        best_y = None

        for y in self:
            dot_product = torch.dot(x_k.grad, y - x_k)
            if dot_product > max_dot:
                max_dot = dot_product
                best_y = y

        return best_y

    def update(self, d_k, d_FW, d_AS, s_k, v_k,alpha_k, alpha_max):
        if self.method == "asfw":
            if id(d_k) == id(d_FW):
                if alpha_k == alpha_max: # maximum stepsize = 1
                    self.clear()
                    self[s_k]=1
                else:
                    # la formula generale per s_k: w = (1-alpha_k)*w + alpha_k, e per gli altri: (1-alpha_k)*w
                    if s_k not in self:
                       self[s_k]=0
                    for x in self:
                        self[x] *= (1 - alpha_k)
                    self[s_k] += alpha_k

            #v_k è sempre nella descrizione
            elif id(d_k) == id(d_AS):
                # la formula generale per v_k: w = (1+alpha_k)*w - alpha_k e per gli altri: w = (1+alpha_k)*w. (se alpha_k = alpha_max allora il peso di v_k = 0)
                for x in self:
                    self[x] *= (1 + alpha_k)
                self[v_k] -= alpha_k

                if alpha_max == alpha_k:
                    del self[v_k] #perche qui il peso sarà sempre 0

        elif self.method == "pwfw":
            if s_k not in self:
                self[s_k] = 0
            if alpha_k == alpha_max:
                del self[v_k]
                self[s_k] = self[s_k] + alpha_k
            else:
                self[s_k] = self[s_k] + alpha_k
                self[v_k] = self[v_k] - alpha_k

        '''if self.cl:
            for _, (atom, weight) in list(self.data.items()):
                #print(weight)
                if weight < 1e-6:
                    del self[atom]
                    print("*")'''

def compute_max_stepsize(method,S,d_FW,v_k,d_k):
    if method == "asfw":
        #if torch.equal(d_k,d_FW):
        if id(d_k)==id(d_FW):
            alpha_max = 1
        else:
            alpha_max = S[v_k] / (1 - S[v_k])

        return alpha_max

    elif method=="pwfw":
        alpha_max = S[v_k]
        return alpha_max

    elif method=="fw":
        return 1

def compute_directions(method, x_k, s_k, S ):
    d_FW = s_k - x_k
    d_AS = None
    v_k = None
    d_k = None

    if method == "asfw":
        v_k = S.max_dot(x_k)
        d_AS = x_k - v_k
        if torch.dot(x_k.grad, d_FW) <= torch.dot(x_k.grad, d_AS):
            d_k = d_FW
        else:
            d_k = d_AS

    elif method == "pwfw":
        v_k = S.max_dot(x_k)
        d_AS = x_k - v_k
        d_k = d_FW + d_AS

    elif method == "fw":
        d_k = d_FW

    return d_FW, d_k, d_AS, v_k

def line_search_armijo(f, starting_step_size, d_k, x_k, g_k ,delta = 0.5, gamma=1e-4):
    m = 0
    f_k = f(x_k)
    c = gamma*torch.dot(g_k,d_k)

    while True:
        alpha = delta**m*starting_step_size
        if f(x_k + alpha*d_k) <= f_k + alpha*c:
            return alpha
        m = m + 1

def UFW(method, f: torch.nn.Module, x_0: torch.Tensor, C: Feasible_Set, max_iter, primal_gap_tol, grad_tol):
    S = ActiveSet(x_0,method)
    fs = []
    x_k = x_0

    for k in range(max_iter):
        x_k = x_k.detach()
        x_k.requires_grad_()

        y_k = f(x_k)
        y_k.backward()
        fs.append(y_k.item())

        s_k = C.lm(x_k, grad_tol)

        # x* is in the interior of C
        if s_k.equal(x_k):
            print('gradient convergence')
            return x_k, fs

        d_FW, d_k, d_AS, v_k = compute_directions(method,x_k,s_k,S)

        if torch.dot(x_k.grad, d_FW) > -primal_gap_tol:
            print("fw gap convergence")
            return x_k, fs

        alpha_max = compute_max_stepsize(method, S, d_FW, v_k, d_k)
        alpha_k = line_search_armijo(f, alpha_max, d_k, x_k, x_k.grad, gamma=1e-4)

        S.update(d_k, d_FW, d_AS, s_k, v_k, alpha_k, alpha_max)

        if f(x_k + alpha_k * d_k) == f(x_k):
            print("numerical convergence")
            return x_k, fs

        x_k = x_k + alpha_k * d_k

    print("max num of iterations reached")
    return x_k, fs