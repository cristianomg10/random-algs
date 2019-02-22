import pandas as pd
import numpy as np
import scipy.sparse as sps

def leader_rank(data, links, num_nodes):

    EG1 = pd.DataFrame(np.zeros((num_nodes, 2)))
    EG2 = pd.DataFrame(np.zeros(( num_nodes, 2)))

    for i in range(0, num_nodes):
        EG1.at[i, 0] = num_nodes
        EG1.at[i, 1] = i

        EG2.at[i, 0] = EG1.at[i, 1]
        EG2.at[i, 1] = EG1.at[i, 0]

    data = data.append(EG1)
    data = data.append(EG2)

    x = data.values.tolist()
    tupl0 = np.array([int(i[0]) for i in x])
    tupl1 = np.array([int(i[1]) for i in x])

    P = sps.csr_matrix((np.ones(data.__len__()), (tupl0, tupl1)), dtype=np.float).toarray()
    D_in = sum(P) # in degree
    D_out = sum(P.transpose()) #out degree

    # transition matrix PP
    EE = pd.DataFrame(np.zeros((num_nodes + 1, 2)))
    for j in range(0, num_nodes + 1):
        EE.at[j, 0] = j
        EE.at[j, 1] = 1 / D_out[j]

    x = EE.values.tolist()
    tupl0 = np.array([int(i[0]) for i in x])
    tupl1 = np.array([(i[1]) for i in x])

    D = sps.csr_matrix((tupl1, (tupl0, tupl0)), dtype=np.float).toarray()
    PP = np.dot(D.tolist(), P.tolist())

    # Diffuusion to stable state
    Gd = pd.DataFrame(np.ones((1, num_nodes + 1)))
    Gd.at[0, num_nodes] = 0
    error = 10000
    error_threshold = 0.00002

    step = 1
    while error > error_threshold:
        print(step)
        M = Gd
        Gd = np.dot(PP.transpose(), Gd.transpose())

        a = Gd.tolist()
        b = M.transpose().values

        part1 = abs(np.subtract(a, b))

        c = M.values
        part2 = np.divide(part1, c)

        error = sum(sum(part2)) / (num_nodes + 1)

        Gd = pd.DataFrame(Gd.transpose())
        step += 1

    b = Gd[num_nodes] / num_nodes

    Gd = Gd.transpose() + b

    #Gd = Gd + b
    Gd.at[num_nodes, 0] = 0

    # Write the ranking results: nodes ID & Score
    R = pd.DataFrame(np.zeros((num_nodes, 2)))

    for i in range(0, num_nodes):
        R.at[i, 0] = i
        R.at[i, 1] = Gd.at[i, 0] * -1

    lista = np.array(R.values.tolist())
    lista = lista[np.argsort(lista[:, 1])]

    return lista


arr = [
    [0, 1],
    [0, 3],
    [0, 4],
    [1, 0],
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 0],
    [3, 2],
    [3, 4],
    [3, 5],
    [4, 0],
    [4, 3],
    [5, 3]
]


data = pd.DataFrame(arr)
print(leader_rank(data , 14, 6))
