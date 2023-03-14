from gurobipy import Model, GRB, quicksum
import time
import json


def read_data(file):
    with open(file) as f:
        next(f)
        A = [x for x in next(f).split()]
        nR = len(A) - 1
        K = []
        while not A[0] == ';':
            aux = []
            for r in range(nR):
                aux.append(float(A[r + 1]))
            A = [x for x in next(f).split()]
            K.append(aux)
        nL = len(K)
        next(f)
        next(f)
        D = []
        A = [x for x in next(f).split()]
        while not A[0] == ';':
            D.append(float(A[1]))
            A = [x for x in next(f).split()]
        next(f)
        next(f)
        E = []
        A = [x for x in next(f).split()]
        while not A[0] == ';':
            E.append(float(A[1]))
            A = [x for x in next(f).split()]
        return K, D, E


def modelo_mip(R, L, T, D, E, K, alpha, beta, C):
    M = 1e6
    mdl = Model('diferido')
    start_time = time.time()
    # Variables
    RT = [(r, t) for r in R for t in T]
    LT = [(l, t) for l in L for t in T]
    x = mdl.addVars(RT, vtype=GRB.BINARY)
    u = mdl.addVars(LT, vtype=GRB.BINARY)
    y = mdl.addVars(LT, vtype=GRB.CONTINUOUS, lb=0.0)
    z = mdl.addVars(LT, vtype=GRB.CONTINUOUS, lb=0.0)
    v = mdl.addVars(L, vtype=GRB.CONTINUOUS, lb=0.0)
    # Objective
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(alpha * quicksum(v[l] for l in L) + beta * quicksum(u[l, t] * y[l, t] for l in L for t in T))
    # Constraints
    mdl.addConstrs(quicksum(x[r, t] for t in T) == 1 for r in R)
    mdl.addConstrs(quicksum(K[l - 1][r - 1] * x[r, t] for r in R) == y[l, t] for t in T for l in L)
    mdl.addConstrs(y[l, t] <= z[l, t] for t in T for l in L)
    mdl.addConstrs(z[l, t] <= E[l - 1] + v[l] for t in T for l in L)
    mdl.addConstrs(quicksum(z[l, t] for t in T) <= D[l - 1] for l in L)
    mdl.addConstrs(y[l, t] - C * z[l, t] <= M * u[l, t] for t in T for l in L)
    # Solve
    mdl.Params.MIPGap = 6 / 100
    # mdl.Params.TimeLimit = 3600  # seconds
    mdl.optimize()
    etime = time.time() - start_time
    if not mdl.Status == 3:
        x = [[r, t, x[r, t].x] for r in R for t in T if x[r, t].x > 0.99]
        u = [[l, t, u[l, t].x] for l in L for t in T if u[l, t].x > 0.99]
        y = [[l, t, y[l, t].x] for l in L for t in T]
        z = [[l, t, z[l, t].x] for l in L for t in T]
        v = [[l, v[l].x] for l in L]
        return [x, u, y, z, v, etime]
    else:
        return [0]


K, D, E = read_data('data.txt')
L = range(1, len(K) + 1)
R = range(1, len(K[0]) + 1)
T = range(1, 3 + 1)
alpha, beta = 1, 1


C = 0.75
out = modelo_mip(R, L, T, D, E, K, alpha, beta, C)
dict = {"x": out[0],
        "u": out[1],
        "y": out[2],
        "z": out[3],
        "v": out[4],
        "time": out[5]}
with open(f'results-{C}.json', 'w') as outfile:
    json.dump(dict, outfile, indent=2)
