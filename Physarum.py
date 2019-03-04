import numpy as np


def Physarum(L, source, sink):
    n = len(L[0:])
    D = 0.5 * np.ones([n,n])
    D[L == 0] = 0;
    L[L==0] = float('inf')
    A = np.zeros(n)
    A[source] = 1
    A[sink] = -1
    tempD = np.zeros([n,n])
    while sum(sum(np.abs(tempD-D))) >= 0.001:
        B = D / L
        B = B - np.diag(sum(B))
        P = np.linalg.lstsq(B[:,:-1],A)[0]
        P = np.append(P, [0])
        temp = np.tile(P, [n,1])
        Pmat = temp.T - temp
        Q = (D / L) * Pmat;
        Q = abs(Q);
        tempD = D.copy();
        D = (Q + D) / 2;
    return D
