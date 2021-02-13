# -*- coding: utf-8 -*-

import sys
import time
import argparse
from scipy.linalg import expm
import scipy.linalg as spl
import numpy as np
from msvd import tensor_svd
from msvd import tensor_eigh
from msvd import tensor_QR
from msvd import psvd
from itertools import product

"""
クラス Tensors_CTMに格納されているテンソルの定義


    C1--1     0--T1--2     0--C2
    |            |             |
    0            1             1

    2            0             0          
    |            |             |          
    T4--1     3--A--1     1 --T2       
    |            |             |        
    0            2             2          

    1           1              0
    |           |              | 
    C4--0    2--T3--0      1--C3

PEPSの順番
          0          0            0        0
         /          /             |        |
    3-- a --1   3-- b --1      3--A--1  3--B--1
      / |         / |             |        |
     2  4        2  4             2        2

Isometryの引数の定義

C1 -1       0--T11--2       0--T12--2       0- C2 
|               |1              |1              |
0                                               1

2                                               0
|                                               |
T42-1           A1            A2             1-T21
|                                               |
0                                               2

2                                               0
|                                               |    
T41-1           A4            A3            1-T22
|                                               |
0                                               2
 
1                                               0
|               |1          |1                  |
C4 -0       2--T32--0    2--T31--0           1- C3

  0    1  2
  |    |  |
  P    P_til
 | |    |
 1 2
"""
class Tensors_CTM():

    def __init__(self,ta):

        dim = ta.shape[0]
        Ta = np.tensordot(ta, ta.conj(), ([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        Ta = Ta.reshape(dim**2, dim**2, dim**2, dim**2)

        C1, C2, C3, C4, T1, T2, T3, T4 = initial_CTM(ta)
        self.Ta  = Ta
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self.P  = T4
        self.P_til = T4

        self.mxa=0
        self.mza=0
        self.mxb=0
        self.mzb=0
        self.mxc=0
        self.mzc=0
        self.E_down = 0
################################################################
def spin_operators(S):

    d = int(np.rint(2*S + 1))
    dz = np.zeros(d);  mp = np.zeros(d-1)

    for n in range(d-1):
        dz[n] = S - n
        mp[n] = np.sqrt((2.0*S - n)*(n + 1.0))

        dz[d - 1] = - S
    Sp = np.diag(mp,1);   Sm = np.diag(mp,-1)
    Sx = 0.5*(Sp + Sm);   Sy = -0.5j*(Sp - Sm)
    Sz = np.diag(dz)


    return Sx, Sy, Sz

def Hamiltonian_Heisen_In_Trian(J,Hz,spin):

    Sx, Sy, Sz = spin_operators(spin)
    I =np.eye(d_spin,d_spin)

    H_BC = np.kron(I, np.kron(Sx,Sx)) + np.kron(I, np.kron(Sy,Sy)) + np.kron(I, np.kron(Sz,Sz))
    H_AB = np.kron(np.kron(Sx,Sx), I) + np.kron(np.kron(Sy,Sy), I) + np.kron(np.kron(Sz,Sz), I)
    H_CA = np.kron(np.kron(Sx,I), Sx) + np.kron(np.kron(Sy,I), Sy) + np.kron(np.kron(Sz,I), Sz)

    Ham = J*(np.kron(Sx,Sx) + np.kron(Sy,Sy) + np.kron(Sz,Sz)) - 0.25 * Hz *( np.kron(Sz,I) + np.kron(I,Sz) )
    H =  J*(H_AB + H_BC + H_CA) - 0.5*Hz*(np.kron(np.kron(Sz,I), I) + np.kron(np.kron(I,Sz), I) + np.kron(np.kron(I,I), Sz))
    #print(np.real(H_AB))
    #print(np.real(H_BC))
    #print(np.real(H_CA))
    #print(np.real(H))
    #exit()



    return np.real(H), np.real(Ham.reshape(d_spin, d_spin, d_spin, d_spin))

def Hamiltonian_Chiral_In_Trian(J,J_chi,Hz,spin):

    Sx, Sy, Sz = spin_operators(spin)
    I =np.eye(d_spin,d_spin)

    H_BC = np.kron(I, np.kron(Sx,Sx)) + np.kron(I, np.kron(Sy,Sy)) + np.kron(I, np.kron(Sz,Sz))
    H_AB = np.kron(np.kron(Sx,Sx), I) + np.kron(np.kron(Sy,Sy), I) + np.kron(np.kron(Sz,Sz), I)
    H_CA = np.kron(np.kron(Sx,I), Sx) + np.kron(np.kron(Sy,I), Sy) + np.kron(np.kron(Sz,I), Sz)

    Ham = J*(np.kron(Sx,Sx) + np.kron(Sy,Sy) + np.kron(Sz,Sz)) - 0.25 * Hz *( np.kron(Sz,I) + np.kron(I,Sz) )
    H =  J*(H_AB + H_BC + H_CA) - 0.5*Hz*(np.kron(np.kron(Sz,I), I) + np.kron(np.kron(I,Sz), I) + np.kron(np.kron(I,I), Sz))
    H_chi = J_chi*( np.kron(np.kron(Sx,Sy), Sz) + np.kron(np.kron(Sy,Sz), Sx) + np.kron(np.kron(Sz,Sx), Sy) \
        - np.kron(np.kron(Sx,Sz), Sy) - np.kron(np.kron(Sy,Sx), Sz) - np.kron(np.kron(Sz,Sy), Sx))

    return H_chi + H
###########################################################################
def initial_iPESS(Dx, d_spin):

    ## random
    
    A1 = np.random.random((Dx, d_spin, Dx))# + 1.0j
    B1 = np.random.random((Dx, d_spin, Dx))# + 1.0j 
    C1 = np.random.random((Dx, d_spin, Dx))# + 1.0j

    A2 = np.random.random((Dx, d_spin, Dx))# + 1.0j
    B2 = np.random.random((Dx, d_spin, Dx))# + 1.0j 
    C2 = np.random.random((Dx, d_spin, Dx))# + 1.0j

    A3 = np.random.random((Dx, d_spin, Dx))# + 1.0j
    B3 = np.random.random((Dx, d_spin, Dx))# + 1.0j 
    C3 = np.random.random((Dx, d_spin, Dx))# + 1.0j

    R_up  = np.random.random((Dx,Dx,Dx))# + 1.0j
    R_low = np.random.random((Dx,Dx,Dx))# + 1.0j

   



    # vector lu, lr, ld, ll
    l = np.ones(A1.shape[0], dtype=float)
    for i in np.arange(len(l)):    l[i] /= 10**i
    l /= np.sqrt(np.dot(l,l))
    
    return A1, A1, A3, A1, A3, A1, A3, A1, A1, R_up, R_up, R_up, R_up, R_up, R_up, \
           l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l    

    #return A1, A1, A1, A1, A1, A1, A1, A1, A1, R_up, R_up, R_up, R_up, R_up, R_up, \
    #       l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l  

    #return A1, A2, A3, B1, B2, B3, C1, C2, C3, R_up, R_up, R_up, R_low, R_low, R_low, \
    #       l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l

def SimpleUpdate_down(A,B,C,R,la,lb,lc,U):
    #
    #             0 1
    #  \|     |/   \|
    #   A     B     A     index 0: outgoing
    #    \ | /       \2   index 2: ingoint
    #      R
    #      |          0   1
    #      C/          \R/
    #      |            |
    #                   2
    #
    #

    A = A*la[:,None,None] 
    B = B*lb[:,None,None]
    C = C*lc[:,None,None]

    
    T = np.transpose(
        np.tensordot(
            A, np.tensordot(
                B, np.tensordot(
                    C, R, ([2], [2])
                ), ([2], [3])
            ), ([2], [4])
        ), [0, 2, 4, 1, 3, 5]
    )


    V  = np.tensordot(
        T, U, ([3, 4, 5], [3, 4, 5])
    )


    Tmp = np.tensordot(V, V.conj(),([1,2,4,5], [1,2,4,5]) ) ##  (0,3)
    #uA, la_new = tensor_eigh(Tmp, (0,1),(2,3),D)
    uA, la_new, _ = tensor_svd(Tmp,(0,1),(2,3),D)
    la_new = np.sqrt(la_new)
    la_new = la_new/np.sqrt(np.dot(la_new,la_new))
    A = uA*(1/la)[:,None,None]

    Tmp = np.tensordot(V, V.conj(),([0,2,3,5], [0,2,3,5]) ) ##  (1,4)
    #uB, lb_new = tensor_eigh(Tmp, (0,1),(2,3),D)
    uB, lb_new, _ = tensor_svd(Tmp,(0,1),(2,3),D)
    lb_new = np.sqrt(lb_new)
    lb_new = lb_new/np.sqrt(np.dot(lb_new,lb_new))
    B = uB*(1/lb)[:,None,None] 

    Tmp = np.tensordot(V, V.conj(),([0,1,3,4], [0,1,3,4]) ) ##  (2,5)
    #uC, lc_new = tensor_eigh(Tmp, (0,1),(2,3),D)
    uC, lc_new, _ = tensor_svd(Tmp,(0,1),(2,3),D)
    lc_new = np.sqrt(lc_new)
    lc_new = lc_new/np.sqrt(np.dot(lc_new,lc_new))
    C = uC*(1/lc)[:,None,None] 

    
    R_new = np.tensordot(
        uA.conj(), np.tensordot(
            uB.conj(), np.tensordot(
                V, uC.conj(), ([2, 5], [0, 1])
            ), ([0, 1], [1, 3])
        ), ([0, 1], [1, 2])
    )

    R_new /=np.max(abs(R_new))

  


    return A, B, C, R_new, la_new, lb_new, lc_new

def SimpleUpdate_up(B,C,A,R,lb,lc,la,U):
    #
 
    B =B*lb[None,None,:]
    C =C*lc[None,None,:]
    A =A*la[None,None,:]

    T = np.transpose(
        np.tensordot(
            B, np.tensordot(
                C, np.tensordot(
                    A, R, ([0], [2])
                ), ([0], [3])
            ), ([0], [4])
        ), [1, 3, 5, 0, 2, 4]
    )

    V  = np.tensordot(
        T, U, ([3, 4, 5], [3, 4, 5])
    )

    Tmp = np.tensordot(V, V.conj(),([1,2,4,5], [1,2,4,5]) ) ##  (0,3)
    uB, lb_new, _ = tensor_svd(Tmp,(0,1),(2,3),D)
    lb_new = np.sqrt(lb_new)
    lb_new = lb_new/np.sqrt(np.dot(lb_new,lb_new))
    B = np.transpose(uB*(1/lb)[:,None,None],[2,1,0])

    Tmp = np.tensordot(V, V.conj(),([0,2,3,5], [0,2,3,5]) ) ##  (1,4)
    uC, lc_new, _ = tensor_svd(Tmp,(0,1),(2,3),D)
    lc_new = np.sqrt(lc_new)
    lc_new = lc_new/np.sqrt(np.dot(lc_new,lc_new))
    C = np.transpose(uC*(1/lc)[:,None,None],[2,1,0]) 

    Tmp = np.tensordot(V, V.conj(),([0,1,3,4], [0,1,3,4]) ) ##  (2,5)
    uA, la_new, _ = tensor_svd(Tmp,(0,1),(2,3),D)
    la_new = np.sqrt(la_new)
    la_new = la_new/np.sqrt(np.dot(la_new,la_new))
    A = np.transpose(uA*(1/la)[:,None,None] ,[2,1,0])

    R_new = np.tensordot(
        uB.conj(), np.tensordot(
            uC.conj(), np.tensordot(
                V, uA.conj(), ([2, 5], [0, 1])
            ), ([0, 1], [1, 3])
        ), ([0, 1], [1, 2])
    )
    R_new /=np.max(abs(R_new))



    return B, C, A, R_new, lb_new, lc_new, la_new

###########################################################################

def Energy_Triangle(A,B,C,R, la_up, lb_up, lc_up, H, Ham):

    A = A*la_up[:,None,None] ; B = B*lb_up[:,None,None]; C = C*lc_up[:,None,None]
    H = H.reshape(d_spin, d_spin, d_spin, d_spin, d_spin, d_spin)

    """
    tmp = np.transpose(
        np.tensordot(
            np.tensordot(
                A, A.conj(), ([0], [0])
            ), np.tensordot(
                np.tensordot(
                    B, np.tensordot(
                        B.conj(), R, ([2], [1])
                    ), ([0], [0])
                ), np.tensordot(
                    C.conj(), np.tensordot(
                        C, R.conj(), ([2], [2])
                    ), ([0], [0])
                ), ([1, 4], [4, 1])
            ), ([1, 3], [5, 2])
        ), [0, 2, 5, 1, 3, 4]
    )
    
    """
    tmp = np.transpose(
        np.tensordot(
            np.tensordot(
                A, A.conj(), ([0], [0])
            ), np.tensordot(
                np.tensordot(
                    B, np.tensordot(
                        C, R, ([2], [2])
                    ), ([2], [3])
                ), np.tensordot(
                    B.conj(), np.tensordot(
                        C.conj(), R.conj(), ([2], [2])
                    ), ([2], [3])
                ), ([0, 2], [0, 2])
            ), ([1, 3], [2, 5])
        ), [0, 2, 3, 1, 4, 5]
    )
    
    norm = np.einsum(tmp, (0, 1, 2, 0, 1, 2), ())

    E_AB = np.einsum(tmp, (0, 1, 2, 3, 4, 2), (0, 1, 3, 4)) 
    E_AB = np.tensordot(E_AB, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E_BC = np.einsum(tmp, (0, 1, 2, 0, 3, 4), (1, 2, 3, 4))
    E_BC = np.tensordot(E_BC, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E_CA = np.einsum(tmp, (0, 1, 2, 3, 1, 4), (0, 2, 3, 4))
    E_CA = np.tensordot(E_CA, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E = np.tensordot(tmp.conj(), H, ([0,1,2,3,4,5],[0,1,2,3,4,5]))/norm

    #print(E, E_AB+E_BC+E_CA, "\n")
    #print(E_AB, E_BC, E_CA)

    return np.real(E)#, E_AB, E_BC, E_CA

def Magnetization(A, la_up, la_low):

    Sx,Sy,Sz=spin_operators(spin)

    I = np.eye(d_spin, d_spin)
    A = A*la_up[:,None,None]*la_low[None,None,:]

    mz = np.tensordot(
        Sz, np.tensordot(
            A, A.conj(), ([0, 2], [0, 2])
        ), ([0, 1], [0, 1])
    )

    my = np.tensordot(
        Sy, np.tensordot(
            A, A.conj(), ([0, 2], [0, 2])
        ), ([0, 1], [0, 1])
    )

    mx = np.tensordot(
        Sx, np.tensordot(
            A, A.conj(), ([0, 2], [0, 2])
        ), ([0, 1], [0, 1])
    )




    return mx, my, mz

###########################################################################
def Calcu_Unit_down(A,B,C,R_up,R_low):

        psi = np.transpose(
            np.tensordot(
                np.tensordot(
                    C, R_up, ([0], [1])
                ), np.tensordot(
                    A, np.tensordot(
                        B, R_low, ([2], [1])
                    ), ([2], [2])
                ), ([1], [4])
            ), [3, 5, 2, 1, 4, 6, 0]
        )

        return psi

def initial_CTM(ta):

    dim = ta.shape[0]*ta.shape[0]
    C1 = np.transpose(np.tensordot(ta, ta.conj(), ([0,3,4,5,6],[0,3,4,5,6])), (1,3,0,2))
    C1 = C1.reshape(dim,dim)

    C2 = np.transpose(np.tensordot(ta, ta.conj(), ([0,1,4,5,6],[0,1,4,5,6])), (1,3,0,2))
    C2 = C2.reshape(dim,dim)

    C3 = np.transpose(np.tensordot(ta, ta.conj(), ([1,2,4,5,6],[1,2,4,5,6])), (0,2,1,3))
    C3 = C3.reshape(dim,dim)

    C4 = np.transpose(np.tensordot(ta, ta.conj(), ([2,3,4,5,6],[2,3,4,5,6])), (1,3,0,2))
    C4 = C4.reshape(dim,dim)

    T1 = np.transpose(np.tensordot(ta, ta.conj(), ([0,4,5,6],[0,4,5,6])), (2,5,1,4,0,3))
    T1 = T1.reshape(dim,dim,dim)

    T2 = np.transpose(np.tensordot(ta, ta.conj(), ([1,4,5,6],[1,4,5,6])), (0,3,2,5,1,4))
    T2 = T2.reshape(dim,dim,dim)

    T3 = np.transpose(np.tensordot(ta, ta.conj(), ([2,4,5,6],[2,4,5,6])), (1,4,0,3,2,5))
    T3 = T3.reshape(dim,dim,dim)

    T4 = np.transpose(np.tensordot(ta, ta.conj(), ([3,4,5,6],[3,4,5,6])), (2,5,1,4,0,3))
    T4 = T4.reshape(dim,dim,dim)

    return C1, C2, C3, C4, T1, T2, T3, T4

def ComputeQuantities_down_triangle(ta,Ta,A2,A3,A4,C1,C2,C3,C4,T11,T12,T21,T22,T31,T32,T41,T42,H):

    def Calc_impurity_tensor(psi):

        Sx,Sy,Sz = spin_operators(spin)

        ## Energy on ABC
        H_psi = np.tensordot(psi, H.reshape(d_spin, d_spin, d_spin, d_spin, d_spin, d_spin), ([4,5,6],[3,4,5]))
        Ta_imp = np.tensordot(H_psi.conj(),psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        dim = Ta_imp.shape[0]
        Ta_imp = Ta_imp.reshape(dim**2, dim**2, dim**2, dim**2)

        # Magnetization on A site
        SzA_psi = np.tensordot(psi, Sz, ([4],[1])).transpose(0,1,2,3,6,4,5)
        SzA_imp = np.tensordot(SzA_psi.conj(),psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SzA_imp = SzA_imp.reshape(dim**2, dim**2, dim**2, dim**2)
        SxA_psi = np.tensordot(psi, Sx, ([4],[1])).transpose(0,1,2,3,6,4,5)
        SxA_imp = np.tensordot(SxA_psi.conj(),psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SxA_imp = SxA_imp.reshape(dim**2, dim**2, dim**2, dim**2)

        # Magnetization on B site
        SzB_psi = np.tensordot(psi, Sz, ([5],[1])).transpose(0,1,2,3,4,6,5)
        SzB_imp = np.tensordot(SzB_psi.conj(),psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SzB_imp = SzB_imp.reshape(dim**2, dim**2, dim**2, dim**2)
        SxB_psi = np.tensordot(psi, Sx, ([5],[1])).transpose(0,1,2,3,4,6,5)
        SxB_imp = np.tensordot(SxB_psi.conj(),psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SxB_imp = SxB_imp.reshape(dim**2, dim**2, dim**2, dim**2)

        # Magnetization on C site
        SzC_psi = np.tensordot(psi, Sz, ([6],[1]))
        SzC_imp = np.tensordot(SzC_psi,psi.conj(),([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SzC_imp = SzC_imp.reshape(dim**2, dim**2, dim**2, dim**2)
        SxC_psi = np.tensordot(psi, Sx, ([6],[1]))
        SxC_imp = np.tensordot(SxC_psi,psi.conj(),([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SxC_imp = SxC_imp.reshape(dim**2, dim**2, dim**2, dim**2)

        return Ta_imp, SxA_imp, SzA_imp, SxB_imp, SzB_imp, SxC_imp, SzC_imp

    E_imp, SxA_imp, SzA_imp, SxB_imp, SzB_imp, SxC_imp, SzC_imp = Calc_impurity_tensor(ta)

    Cc1 = np.tensordot(T11, np.tensordot(C1,T42,([0],[2])), ([0],[0]))

    Cc2 = np.transpose(
        np.tensordot(
            A2, np.tensordot(
                T21, np.tensordot(
                    C2, T12, ([0], [2])
                ), ([0], [0])
            ), ([0, 1], [3, 0])
        ), [3, 1, 2, 0]
    )

    Cc3 = np.transpose(
        np.tensordot(
            A3, np.tensordot(
                T31, np.tensordot(
                    C3, T22, ([0], [2])
                ), ([0], [0])
            ), ([1, 2], [3, 0])
        ), [3, 0, 2, 1]
    )
    Cc4 = np.transpose(
        np.tensordot(
            A4, np.tensordot(
                T41, np.tensordot(
                    C4, T32, ([0], [2])
                ), ([0], [0])
            ), ([2, 3], [3, 0])
        ), [3, 1, 2, 0]
    )

    Env = np.transpose(
        np.tensordot(
            Cc1, np.tensordot(
                Cc2, np.tensordot(
                    Cc3, Cc4, ([2, 3], [0, 1])
                ), ([2, 3], [0, 1])
            ), ([1, 2], [0, 2])
        ), [0, 2, 3, 1]
    )

    norm = np.tensordot(Env, Ta, ([0,1,2,3],[0,1,2,3]))
    E = np.tensordot(Env, E_imp, ([0,1,2,3],[0,1,2,3]))/norm

    mxa = np.tensordot(Env, SxA_imp, ([0,1,2,3],[0,1,2,3]))/norm
    mxb = np.tensordot(Env, SxB_imp, ([0,1,2,3],[0,1,2,3]))/norm
    mxc = np.tensordot(Env, SxC_imp, ([0,1,2,3],[0,1,2,3]))/norm
    mza = np.tensordot(Env, SzA_imp, ([0,1,2,3],[0,1,2,3]))/norm
    mzb = np.tensordot(Env, SzB_imp, ([0,1,2,3],[0,1,2,3]))/norm
    mzc = np.tensordot(Env, SzC_imp, ([0,1,2,3],[0,1,2,3]))/norm

    return E, mxa, mza, mxb, mzb, mxc, mzc
###########################################################################

def Isometry(C1_n, C2_n, C3_n, C4_n):
    """
       3
       |
    2--A--0  必ず'move'方向の反対側が'0'になるようにする!
       |
       1
    """


    # upper halfとlower halfを作り、QR分解する
    upper_half = np.tensordot(C2_n, C1_n, ([0,1],[2,3]) )
    lower_half = np.tensordot(C3_n, C4_n, ([2,3],[0,1]) )
    #print( upper_half.shape, lower_half.shape )
    _,R_up = tensor_QR(upper_half, (0,1), (2,3), C1_n.shape[0]*C1_n.shape[1])
    _,R_low = tensor_QR(lower_half, (0,1), (2,3), C1_n.shape[0]*C1_n.shape[1])

    ## Projection Operatorを作る
    U,s,Vdag =psvd(np.tensordot(R_up,R_low, ( [1,2],[1,2]) ) ,chi)
    #s = s/np.sqrt(np.dot(s,s))
    U = U*(np.sqrt(1./s))[None,:]
    Vdag = Vdag*(np.sqrt(1./s))[:,None]
    P = np.tensordot(U,R_up,([0],[0]) )
    P_til = np.tensordot(Vdag,R_low,([1],[0]) )
    #print( np.tensordot(P,P_til, [(1,2),(1,2)]),"\n")

    return P, P_til

def CTM_corner(C1,C2,C3,C4,T11,T12,T21,T22,T31,T32,T41,T42,A1,A2,A3,A4):

    ## それぞれの四隅にまとめる
    ## テンソルの足0が右側を向くように順番が変えられていることに注意
    ## PEPSの順番
    ##    3        
    ##    |        
    ## 2--A--0  
    ##    |  
    ##    1 
    
    Cc1 = np.transpose(
        np.tensordot(
            A1, np.tensordot(
                T11, np.tensordot(
                    C1, T42, ([0], [2])
                ), ([0], [0])
            ), ([2, 3], [3, 0])
        ), [3, 1, 2, 0]
    )

    Cc2 = np.transpose(
        np.tensordot(
            A2, np.tensordot(
                T21, np.tensordot(
                    C2, T12, ([0], [2])
                ), ([0], [0])
            ), ([0, 3], [0, 3])
        ), [3, 1, 2, 0]
    )

    Cc3 = np.transpose(
        np.tensordot(
            A3, np.tensordot(
                T31, np.tensordot(
                    C3, T22, ([0], [2])
                ), ([0], [0])
            ), ([0, 1], [3, 0])
        ), [3, 1, 2, 0]
    )
    Cc4 = np.transpose(
        np.tensordot(
            A4, np.tensordot(
                T41, np.tensordot(
                    C4, T32, ([0], [2])
                ), ([0], [0])
            ), ([1, 2], [3, 0])
        ), [3, 0, 2, 1]
    )

    return Cc1, Cc2, Cc3, Cc4

def CTM_update(C1,C4,T1,T4,T3,P,P_til,A):

    C1_new = np.transpose(
        np.tensordot(
            T1, np.tensordot(
                C1, P_til, ([0], [1])
            ), ([0, 1], [0, 2])
        ), [1, 0]
    )
    C1_new = C1_new/np.amax( np.abs(C1_new) )

    C4_new = np.transpose(
        np.tensordot(
            T3, np.tensordot(
                C4, P, ([1], [1])
            ), ([1, 2], [2, 0])
        ), [0, 1]
    )
    C4_new = C4_new/np.amax( np.abs(C4_new) )

    T4_new = np.transpose(
        np.tensordot(
            P, np.tensordot(
                A, np.tensordot(
                    T4, P_til, ([0], [1])
                ), ([1, 2], [3, 0])
            ), ([1, 2], [2, 1])
        ), [2, 1, 0]
    )
    T4_new = T4_new/np.amax( np.abs(T4_new) )

    return C1_new, C4_new, T4_new

def CTMRG(CTMs, Lx, Ly):
    ######## Left Move ###########
    for x in range(Lx):
        x1 = (x+1)%Lx ; x2 = (x+2)%Lx ; x3 = (x+3)%Lx
        for y in range(Ly):
            y1 = (y+1)%Ly ; y2 = (y+2)%Ly ; y3 = (y+3)%Ly
            C1_n, C2_n, C3_n, C4_n = CTM_corner(CTMs[x][y].C1, CTMs[x3][y].C2, CTMs[x3][y3].C3, CTMs[x][y3].C4, \
                CTMs[x1][y].T1, CTMs[x2][y].T1, CTMs[x3][y1].T2, CTMs[x3][y2].T2,\
                CTMs[x2][y3].T3, CTMs[x1][y3].T3, CTMs[x][y2].T4, CTMs[x][y1].T4,\
                CTMs[x1][y1].Ta.transpose(1,2,3,0) ,\
                CTMs[x2][y1].Ta.transpose(1,2,3,0) ,\
                CTMs[x2][y2].Ta.transpose(1,2,3,0) ,\
                CTMs[x1][y2].Ta.transpose(1,2,3,0))
            CTMs[x][y1].P , CTMs[x][y1].P_til = Isometry(C1_n, C2_n, C3_n, C4_n)

        for y in range(Ly):
            x1 = (x+1)%Lx ; y1 = (y-1)%Ly

            CTMs[x1][y].C1, CTMs[x1][y].C4, CTMs[x1][y].T4 \
            = CTM_update(CTMs[x][y].C1, \
                CTMs[x][y].C4, CTMs[x1][y].T1, CTMs[x][y].T4,\
                CTMs[x1][y].T3, CTMs[x][y1].P, CTMs[x][y].P_til, CTMs[x1][y].Ta.transpose(1,2,3,0))



    
    ######## Up Move #############
    for y in range(Ly):
        y1 = (y+1)%Ly ; y2 = (y+2)%Ly ; y3 = (y+3)%Ly
        for x in range(Lx):
            x1 = (x+1)%Lx ; x2 = (x+2)%Lx; x3 = (x+3)%Lx
            C2_n, C3_n, C4_n, C1_n = CTM_corner(CTMs[x3][y].C2, CTMs[x3][y3].C3, CTMs[x][y3].C4, CTMs[x][y].C1, \
                CTMs[x3][y1].T2, CTMs[x3][y2].T2, CTMs[x2][y3].T3, CTMs[x1][y3].T3,\
                CTMs[x][y2].T4, CTMs[x][y1].T4, CTMs[x1][y].T1, CTMs[x2][y].T1,\
                CTMs[x2][y1].Ta.transpose(2,3,0,1) ,\
                CTMs[x2][y2].Ta.transpose(2,3,0,1) ,\
                CTMs[x1][y2].Ta.transpose(2,3,0,1) ,\
                CTMs[x1][y1].Ta.transpose(2,3,0,1))
            CTMs[x2][y].P , CTMs[x2][y].P_til = Isometry(C2_n, C3_n, C4_n, C1_n)
            #print( "UP ", CTMs[x2][y].P.shape, CTMs[x2][y].P_til.shape, (x,y) )
    
        
        for x in range(Lx):
            x3 = (x+3)%Lx ; x4= (x+4)%Lx
            y1 = (y+1)%Ly  
            CTMs[x3][y1].C2, CTMs[x3][y1].C1, CTMs[x3][y1].T1 \
            = CTM_update(CTMs[x3][y].C2, \
                CTMs[x3][y].C1, CTMs[x3][y1].T2, CTMs[x3][y].T1,\
                CTMs[x3][y1].T4, CTMs[x4][y].P, CTMs[x3][y].P_til, CTMs[x3][y1].Ta.transpose(2,3,0,1))


        
    ######## Right Move ##########
    for x in range(Lx):
        x1 = (x+1)%Lx ; x2 = (x+2)%Lx; x3 = (x+3)%Lx
        for y in range(Ly):
            y1 = (y+1)%Ly ; y2 = (y+2)%Ly ; y3 = (y+3)%Ly
            C3_n, C4_n, C1_n, C2_n = CTM_corner(CTMs[x3][y3].C3, CTMs[x][y3].C4, CTMs[x][y].C1, CTMs[x3][y].C2, \
                CTMs[x2][y3].T3, CTMs[x1][y3].T3, CTMs[x][y2].T4, CTMs[x][y1].T4,\
                CTMs[x1][y].T1, CTMs[x2][y].T1, CTMs[x3][y1].T2, CTMs[x3][y2].T2,\
                CTMs[x2][y2].Ta.transpose(3,0,1,2) ,\
                CTMs[x1][y2].Ta.transpose(3,0,1,2) ,\
                CTMs[x1][y1].Ta.transpose(3,0,1,2) ,\
                CTMs[x2][y1].Ta.transpose(3,0,1,2))
            CTMs[x3][y2].P , CTMs[x3][y2].P_til = Isometry(C3_n, C4_n, C1_n, C2_n)
            #print( "Right ", CTMs[x3][y2].P.shape, CTMs[x3][y2].P_til.shape, (x,y) )


        for y in range(Ly):
            x2 = (x+2)%Lx ; x3 = (x+3)%Lx 
            y3 = (y+3)%Ly ; y4 = (y+4)%Ly

            #print(CTMs[x3][y3].C3.shape, CTMs[x2][y3].P_til.shape )
            CTMs[x2][y3].C3, CTMs[x2][y3].C2, CTMs[x2][y3].T2 \
            = CTM_update(CTMs[x3][y3].C3, \
                CTMs[x3][y3].C2, CTMs[x2][y3].T3, CTMs[x3][y3].T2,\
                CTMs[x2][y3].T1, CTMs[x3][y4].P, CTMs[x3][y3].P_til, CTMs[x2][y3].Ta.transpose(3,0,1,2))

    ######## Down Move ###########
    for y in range(Ly):
        y3 = (y+3)%Ly ; y2 = (y+2)%Ly ; y1 = (y+1)%Ly
        for x in range(Lx):
            x1 = (x+1)%Lx ; x2 = (x+2)%Lx ; x3 = (x+3)%Lx
            C4_n, C1_n, C2_n, C3_n = CTM_corner(CTMs[x][y3].C4, CTMs[x][y].C1, CTMs[x3][y].C2, CTMs[x3][y3].C3, \
                CTMs[x][y2].T4, CTMs[x][y1].T4, CTMs[x1][y].T1, CTMs[x2][y].T1,\
                CTMs[x3][y1].T2, CTMs[x3][y2].T2, CTMs[x2][y3].T3, CTMs[x1][y3].T3,\
                CTMs[x1][y2].Ta,\
                CTMs[x1][y1].Ta,\
                CTMs[x2][y1].Ta,\
                CTMs[x2][y2].Ta)
            CTMs[x1][y3].P , CTMs[x1][y3].P_til = Isometry(C4_n, C1_n, C2_n, C3_n)
            #print( "Down ", CTMs[x1][y3].P.shape, CTMs[x1][y3].P_til.shape, (x,y) )    

        for x in range(Lx):
            x1 = (x-1)%Lx 
    
            CTMs[x][y2].C4, CTMs[x][y2].C3, CTMs[x][y2].T3 \
            = CTM_update(CTMs[x][y3].C4, \
                CTMs[x][y3].C3, CTMs[x][y2].T4, CTMs[x][y3].T3,\
                CTMs[x][y2].T2, CTMs[x1][y3].P, CTMs[x][y3].P_til, CTMs[x][y2].Ta)


    ######## Expectation Value ###########

##  main function
if __name__=="__main__":
    # obtain the arguments

    parser = argparse.ArgumentParser(description='',allow_abbrev=False)
    #parser.add_argument("--omp_cores", type=int, default=1,help="number of OpenMP cores")
    parser.add_argument("--D", type=int, default=2, help="Virtual bond dimension")
    parser.add_argument("--J", type=float, default=1, help="maximal number of epochs")
    parser.add_argument("--dt", type=float, default=0.01, help="inmaginary time")
    parser.add_argument("--chi", type=int, default=20, help="bond dimensions of CTM")
    parser.add_argument("--spin", type=float, default=0.5, help="spin value")
    parser.add_argument("--Hz_start", type=float, default=0., help="intiail value of magnetic field")
    parser.add_argument("--Hz_end", type=float, default=3.0, help="intiail value of magnetic field")
    parser.add_argument("--maxstepTEBD", type=int, default=10000, help="maximal number of TEBD iterations")
    parser.add_argument("--maxstepCTM", type=int, default=10, help="maximal number ofCTM iterations")
    parser.add_argument("--eps_TEBD", type=float, default=1e-9, help="TEBD criterion for convergence")

    args = parser.parse_args()

    D= args.D
    J= args.J
    dt= args.dt
    chi= args.chi
    spin = args.spin
    Hz_start= args.Hz_start
    Hz_end= args.Hz_end
    maxstepTEBD= args.maxstepTEBD
    maxstepCTM= args.maxstepCTM
    d_spin = int(2*spin + 1 )
    tau = dt
    temp = 0.0

    # open the text file
    name = 'D'+str(D)+'S'+str(int(2*spin+1))+'-kagome_iso'
    f = open(name+'.txt','w')

    # criterion for convergence
    eps_TEBD = 1e-9;  eps_CTM = 10**(-10)
    
    # Heisenberg
    H, Ham = Hamiltonian_Heisen_In_Trian(J,Hz_start,spin)
    U = expm(-dt*H).reshape(d_spin, d_spin, d_spin, d_spin, d_spin, d_spin)

    # chiral
    #H = Hamiltonian_Chiral_In_Trian(J,J_chi,0.0,spin)
    #U = expm(-dt*H).reshape(d_spin, d_spin, d_spin, d_spin, d_spin, d_spin)


    # intiail iPESS
    A1, A2, A3, B1, B2, B3, C1, C2, C3,\
    R1_up, R2_up, R3_up, R1_low, R2_low, R3_low,\
    l_A1_up, l_A2_up, l_A3_up,\
    l_A1_low, l_A2_low, l_A3_low,\
    l_B1_up, l_B2_up, l_B3_up,\
    l_B1_low, l_B2_low, l_B3_low,\
    l_C1_up, l_C2_up, l_C3_up,\
    l_C1_low, l_C2_low, l_C3_low, =initial_iPESS(D,d_spin)

    for Hz in np.arange(Hz_start, Hz_end, 0.1):
        tau=dt
        for i in range(maxstepTEBD):

            H, Ham = Hamiltonian_Heisen_In_Trian(J,Hz,spin)
            H1, Ham1 = Hamiltonian_Heisen_In_Trian(J,Hz,spin)
            U = expm(-tau*H).reshape(d_spin, d_spin,d_spin, d_spin, d_spin, d_spin)
            U1 = expm(-tau*H1).reshape(d_spin, d_spin,d_spin, d_spin, d_spin, d_spin)

            #H = Hamiltonian_Chiral_In_Trian(J,J_chi,0.0,spin)
            #U = expm(-tau*H).reshape(d_spin, d_spin, d_spin, d_spin, d_spin, d_spin)

            
            A1, B1, C1, R1_low, l_A1_low, l_B1_low, l_C1_low = \
            SimpleUpdate_down(A1,B1,C1,R1_low,l_A1_up,l_B1_up,l_C1_up,U)
            
            A2, B2, C2, R2_low, l_A2_low, l_B2_low, l_C2_low = \
            SimpleUpdate_down(A2,B2,C2,R2_low,l_A2_up,l_B2_up,l_C2_up,U)

            A3, B3, C3, R3_low, l_A3_low, l_B3_low, l_C3_low = \
            SimpleUpdate_down(A3,B3,C3,R3_low,l_A3_up,l_B3_up,l_C3_up,U) 
            
            
            B1, C2, A3, R2_up, l_B1_up, l_C2_up, l_A3_up = \
            SimpleUpdate_up(B1,C2,A3,R2_up,l_B1_low,l_C2_low,l_A3_low,U1)

            B2, C3, A1, R3_up, l_B2_up, l_C3_up, l_A1_up = \
            SimpleUpdate_up(B2,C3,A1,R3_up,l_B2_low,l_C3_low,l_A1_low,U1)

            B3, C1, A2, R1_up, l_B3_up, l_C1_up, l_A2_up = \
            SimpleUpdate_up(B3,C1,A2,R1_up,l_B3_low,l_C1_low,l_A2_low,U1)
            
            
            #B1, C1, A1, R1_up, l_B1_up, l_C1_up, l_A1_up = \
            #SimpleUpdate_up(B1,C1,A1,R1_up,l_B1_low,l_C1_low,l_A1_low,U)         

            ##  Calculate Energy
            E1_down = Energy_Triangle(A1,B1,C1,R1_low, l_A1_up, l_B1_up, l_C1_up, H, Ham)
            E1_up   = Energy_Triangle(B1.transpose(2,1,0), C2.transpose(2,1,0), A3.transpose(2,1,0), R2_up, l_B1_low, l_C2_low, l_A3_low, H1, Ham1)
            #E1_up   = Energy_Triangle(B1.transpose(2,1,0), C1.transpose(2,1,0), A1.transpose(2,1,0), R1_up, l_B1_low, l_C1_low, l_A1_low, H, Ham)

            E2_down = Energy_Triangle(A2,B2,C2,R2_low, l_A2_up, l_B2_up, l_C2_up, H, Ham)
            E2_up   = Energy_Triangle(B2.transpose(2,1,0), C3.transpose(2,1,0), A1.transpose(2,1,0), R3_up, l_B2_low, l_C3_low, l_A1_low, H1, Ham1)

            E3_down = Energy_Triangle(A3,B3,C3,R3_low, l_A3_up, l_B3_up, l_C3_up, H, Ham)
            E3_up   = Energy_Triangle(B3.transpose(2,1,0), C1.transpose(2,1,0), A2.transpose(2,1,0), R1_up, l_B3_low, l_C1_low, l_A2_low, H1, Ham1)

            E1 = (E1_up + E1_down)/3.  
            E2 = (E2_up + E2_down)/3.
            E3 = (E3_up + E3_down)/3.

            ## Calculate Magnetization
            mxa1, mya1, mza1 = Magnetization(A1, l_A1_up, l_A1_low)
            mxb1, myb1, mzb1 = Magnetization(B1, l_B1_up, l_B1_low)
            mxc1, myc1, mzc1 = Magnetization(C1, l_C1_up, l_C1_low)

            mxa2, mya2, mza2 = Magnetization(A2, l_A2_up, l_A2_low)
            mxb2, myb2, mzb2 = Magnetization(B2, l_B2_up, l_B2_low)
            mxc2, myc2, mzc2 = Magnetization(C2, l_C2_up, l_C2_low)

            mxa3, mya3, mza3 = Magnetization(A3, l_A3_up, l_A3_low)
            mxb3, myb3, mzb3 = Magnetization(B3, l_B3_up, l_B3_low)
            mxc3, myc3, mzc3 = Magnetization(C3, l_C3_up, l_C3_low)

            Mz =(mza1 + mza2 + mza3 + mzb1 + mzb2 + mzb3 + mzc1 + mzc2 + mzc3)/9. 


            if i%100 ==0:
                print(i, Mz/spin, abs(temp-Mz), (E1+E2+E3)/3.0)
                print(i, E1, E2, E3,"\n")
                #print(i, l_A1_up)
                #print(i, l_A1_low)
                #print(i, l_B1_up)
                #print(i, l_B1_low)

                if abs(temp-Mz)<eps_TEBD and tau!=0.1 and tau!=0.01:
                    break
                else: temp = Mz
            if (i+1)%3000 ==0:
                tau = max(tau/10,0.00001)

          
        print("A1: ",mxa1, mza1, np.sqrt(mxa1**2 + mza1**2))
        print("B1: ",mxb1, mzb1, np.sqrt(mxb1**2 + mzb1**2))
        print("C1: ",mxc1, mzc1, np.sqrt(mxc1**2 + mzc1**2), "\n")

        print("A2: ",mxa2, mza2, np.sqrt(mxa2**2 + mza2**2))
        print("B2: ",mxb2, mzb2, np.sqrt(mxb2**2 + mzb2**2))
        print("C2: ",mxc2, mzc2, np.sqrt(mxc2**2 + mzc2**2), "\n")

        print("A3: ",mxa3, mza3, np.sqrt(mxa3**2 + mza3**2))
        print("B3: ",mxb3, mzb3, np.sqrt(mxb3**2 + mzb3**2))
        print("C3: ",mxc3, mzc3, np.sqrt(mxc3**2 + mzc3**2), "\n")
        
        print(Hz, Mz/spin)
        f.write("{0:.8e}, {1:.8e}, {2:.8e}, {3:.8e}, {4:.8e}, {5:.8e}, {6:.8e}, {7:.8e}, {8:.8e}, {9:.8e}, {10:.8e}, {11:.8e}, {12:.8e}, {13:.8e}, {14:.8e}, {15:.8e}, {16:.8e}, {17:.8e}, {18:.8e}, {19:.8e}\n"\
            .format(Hz, Mz/spin, mza1, mzb1, mzc1, mxa1, mxb1, mxc1, mza2, mzb2, mzc2, mxa2, mxb2, mxc2, mza3, mzb3, mzc3, mxa3, mxb3, mxc3) )

        f.flush() 
        #ta1 = Calcu_Unit_down(A1,B1,C1, R1_up, R1_low)
        #ta2 = Calcu_Unit_down(A2,B2,C2, R2_up, R2_low)
        #ta3 = Calcu_Unit_down(A3,B3,C3, R3_up, R3_low)
    
        



        ####### CTMRG ###############
        """
        

        Lx=3; Ly=3
        SU_ten =[ [ta1,ta2,ta3], [ta2,ta3,ta1],[ta3,ta1,ta2] ]
        CTMs = [[0 for y in range(Ly)]  for x in range(Lx)  ]


        ## SUで得られた初期テンソルをクラスのリストに代入する
        for x, y in product(range(Lx), range(Ly) ):
            tensor = Tensors_CTM(SU_ten[y][x])
            CTMs[x][y] = tensor

        for i in range(maxstepCTM):
            Mz_tot = 0.0
            CTMRG(CTMs, Lx, Ly)

            for x, y in product(range(Lx), range(Ly) ):
                x1 = (x+1)%Lx ; x2 = (x+2)%Lx ; x3 = (x+3)%Lx 
                y1 = (y+1)%Ly ; y2 = (y+2)%Ly ; y3 = (y+3)%Ly

                E_down, mxa, mza, mxb, mzb, mxc, mzc= ComputeQuantities_down_triangle(\
                    SU_ten[y1][x1], CTMs[x1][y1].Ta, CTMs[x2][y1].Ta, CTMs[x2][y2].Ta, CTMs[x1][y2].Ta,\
                    CTMs[x][y].C1, CTMs[x2][y].C2, CTMs[x2][y2].C3, CTMs[x][y2].C4,\
                    CTMs[x1][y].T1, CTMs[x2][y1].T1, CTMs[x3][y2].T2, CTMs[x3][y2].T2,\
                    CTMs[x2][y3].T3, CTMs[x1][y3].T3, CTMs[x][y2].T4, CTMs[x][y1].T4,H)

                CTMs[x1][y1].mxa=mxa
                CTMs[x1][y1].mza=mza
                CTMs[x1][y1].mxb=mxb
                CTMs[x1][y1].mzb=mzb
                CTMs[x1][y1].mxc=mxc
                CTMs[x1][y1].mzc=mzc
                CTMs[x1][y1].E_down=E_down

                #print(i, x1,y1, E_down)
                #print("A site: ",mxa, mza, np.sqrt(mxa**2 + mza**2))
                #print("B site: ",mxb, mzb, np.sqrt(mxb**2 + mzb**2))
                #print("C site: ",mxc, mzc, np.sqrt(mxc**2 + mzc**2),"\n")
            print(i,E_down)
            Mza = CTMs[0][0].mza + CTMs[1][0].mza + CTMs[1][1].mza
            Mzb = CTMs[0][0].mzb + CTMs[1][0].mzb + CTMs[1][1].mzb
            Mzc = CTMs[0][0].mzc + CTMs[1][0].mzc + CTMs[1][1].mzc
            Mz_tot = (Mza + Mzb + Mzc)/9.

        print("CTMRG Result")
        print("E1_down: ", CTMs[0][0].E_down) 
        print("A1: ", CTMs[0][0].mxa, CTMs[0][0].mza)
        print("B1: ", CTMs[0][0].mxb, CTMs[0][0].mzb)
        print("C1: ", CTMs[0][0].mxc, CTMs[0][0].mzc,"\n")

        print("E2_down: ", CTMs[2][2].E_down)
        print("A2: ", CTMs[2][2].mxa, CTMs[2][2].mza)
        print("B2: ", CTMs[2][2].mxb, CTMs[2][2].mzb)
        print("C2: ", CTMs[2][2].mxc, CTMs[2][2].mzc,"\n")

        print("E3_down: ", CTMs[1][1].E_down)
        print("A3: ", CTMs[1][1].mxa, CTMs[1][1].mza)
        print("B3: ", CTMs[1][1].mxb, CTMs[1][1].mzb)
        print("C3: ", CTMs[1][1].mxc, CTMs[1][1].mzc,"\n")

        #print(Hz, Mz_tot/spin)
        print( 2*(CTMs[0][0].E_down + CTMs[2][2].E_down + CTMs[1][1].E_down)/9. )
        

        
        print("SU Result")
        print("E1_down: ", E1_down)       
        print("A1 site: ",mxa1, mza1, np.sqrt(mxa1**2 + mza1**2))
        print("B1 site: ",mxb1, mzb1, np.sqrt(mxb1**2 + mzb1**2))
        print("C1 site: ",mxc1, mzc1, np.sqrt(mxc1**2 + mzc1**2),"\n")

        print("E2_down: ", E2_down)
        print("A2 site: ",mxa2, mza2, np.sqrt(mxa2**2 + mza2**2))
        print("B2 site: ",mxb2, mzb2, np.sqrt(mxb2**2 + mzb2**2))
        print("C2 site: ",mxc2, mzc2, np.sqrt(mxc2**2 + mzc2**2),"\n")

        print("E3_down: ", E3_down)
        print("A3 site: ",mxa3, mza3, np.sqrt(mxa3**2 + mza3**2))
        print("B3 site: ",mxb3, mzb3, np.sqrt(mxb3**2 + mzb3**2))
        print("C3 site: ",mxc3, mzc3, np.sqrt(mxc3**2 + mzc3**2),"\n")
            
        Mz =(mza1 + mza2 + mza3 + mzb1 + mzb2 + mzb3 + mzc1 + mzc2 + mzc3)/9. 
        
        
        #print(Hz, Mz/spin)
        print( 2*(E1_down+E2_down+E3_down)/9.)
        exit()
        #f.write("{0:.8e}, {1:.8e}\n".format(Hz, Mz/spin/12.))
        """
    f.close()
        
        
    
    

        
        
        
            
            
          

























