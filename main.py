# -*- coding: utf-8 -*-

import sys
import time
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

        C1, C2, C3, C4, T1, T2, T3, T4 = initial_CTM(Ta)
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


    Tmp = np.tensordot(V, V,([1,2,4,5], [1,2,4,5]) ) ##  (0,3)
    #uA, la_new = tensor_eigh(Tmp, (0,1),(2,3),D_cut)
    uA, la_new, _ = tensor_svd(Tmp,(0,1),(2,3),D_cut)
    la_new = np.sqrt(la_new)
    la_new = la_new/np.sqrt(np.dot(la_new,la_new))
    A = uA*(1/la)[:,None,None]

    Tmp = np.tensordot(V, V,([0,2,3,5], [0,2,3,5]) ) ##  (1,4)
    #uB, lb_new = tensor_eigh(Tmp, (0,1),(2,3),D_cut)
    uB, lb_new, _ = tensor_svd(Tmp,(0,1),(2,3),D_cut)
    lb_new = np.sqrt(lb_new)
    lb_new = lb_new/np.sqrt(np.dot(lb_new,lb_new))
    B = uB*(1/lb)[:,None,None] 

    Tmp = np.tensordot(V, V,([0,1,3,4], [0,1,3,4]) ) ##  (2,5)
    #uC, lc_new = tensor_eigh(Tmp, (0,1),(2,3),D_cut)
    uC, lc_new, _ = tensor_svd(Tmp,(0,1),(2,3),D_cut)
    lc_new = np.sqrt(lc_new)
    lc_new = lc_new/np.sqrt(np.dot(lc_new,lc_new))
    C = uC*(1/lc)[:,None,None] 

    
    R_new = np.tensordot(
        uA, np.tensordot(
            uB, np.tensordot(
                V, uC, ([2, 5], [0, 1])
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
    uB, lb_new, _ = tensor_svd(Tmp,(0,1),(2,3),D_cut)
    lb_new = np.sqrt(lb_new)
    lb_new = lb_new/np.sqrt(np.dot(lb_new,lb_new))
    B = np.transpose(uB*(1/lb)[:,None,None],[2,1,0])

    Tmp = np.tensordot(V, V.conj(),([0,2,3,5], [0,2,3,5]) ) ##  (1,4)
    uC, lc_new, _ = tensor_svd(Tmp,(0,1),(2,3),D_cut)
    lc_new = np.sqrt(lc_new)
    lc_new = lc_new/np.sqrt(np.dot(lc_new,lc_new))
    C = np.transpose(uC*(1/lc)[:,None,None],[2,1,0]) 

    Tmp = np.tensordot(V, V.conj(),([0,1,3,4], [0,1,3,4]) ) ##  (2,5)
    uA, la_new, _ = tensor_svd(Tmp,(0,1),(2,3),D_cut)
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
                        B.conj(), R.conj(), ([2], [1])
                    ), ([0], [0])
                ), np.tensordot(
                    C.conj(), np.tensordot(
                        C, R, ([2], [1])
                    ), ([0], [0])
                ), ([1, 4], [4, 1])
            ), ([1, 3], [5, 2])
        ), [0, 2, 5, 1, 3, 4]
    )
    """

    tmp = np.transpose(
        np.tensordot(
            np.tensordot(
                A, A, ([0], [0])
            ), np.tensordot(
                np.tensordot(
                    B, np.tensordot(
                        B, R, ([2], [1])
                    ), ([0], [0])
                ), np.tensordot(
                    C, np.tensordot(
                        C, R, ([2], [2])
                    ), ([0], [0])
                ), ([1, 4], [4, 1])
            ), ([1, 3], [5, 2])
        ), [0, 2, 5, 1, 3, 4]
    )


    norm = np.einsum(tmp, (0, 1, 2, 0, 1, 2), ())

    E_AB = np.einsum(tmp, (0, 1, 2, 3, 4, 2), (0, 1, 3, 4)) 
    E_AB = np.tensordot(E_AB, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E_BC = np.einsum(tmp, (0, 1, 2, 0, 3, 4), (1, 2, 3, 4))
    E_BC = np.tensordot(E_BC, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E_CA = np.einsum(tmp, (0, 1, 2, 3, 1, 4), (0, 2, 3, 4))
    E_CA = np.tensordot(E_CA, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E = np.tensordot(tmp, H, ([0,1,2,3,4,5],[0,1,2,3,4,5]))/norm

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

def initial_CTM(Ta):

    dim = Ta.shape[0]
    C1 = np.random.random((dim, dim))
    T11 = np.ones((dim, dim, dim))
    C1 = C1 + C1.T

    return C1, C1, C1, C1, T11, T11, T11, T11

def ComputeQuantities_down_triangle(ta,Ta,C1,C2,C3,C4,T1,T2,T3,T4,H):

    def Calc_impurity_tensor(psi):

        Sx,Sy,Sz = spin_operators(spin)

        ## Energy on ABC
        H_psi = np.tensordot(psi, H.reshape(d_spin, d_spin, d_spin, d_spin, d_spin, d_spin), ([4,5,6],[3,4,5]))
        Ta_imp = np.tensordot(H_psi,psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        dim = Ta_imp.shape[0]
        Ta_imp = Ta_imp.reshape(dim**2, dim**2, dim**2, dim**2)

        # Magnetization on A site
        SzA_psi = np.tensordot(psi, Sz, ([4],[1])).transpose(0,1,2,3,6,4,5)
        SzA_imp = np.tensordot(SzA_psi,psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SzA_imp = SzA_imp.reshape(dim**2, dim**2, dim**2, dim**2)
        SxA_psi = np.tensordot(psi, Sx, ([4],[1])).transpose(0,1,2,3,6,4,5)
        SxA_imp = np.tensordot(SxA_psi,psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SxA_imp = SxA_imp.reshape(dim**2, dim**2, dim**2, dim**2)

        # Magnetization on B site
        SzB_psi = np.tensordot(psi, Sz, ([5],[1])).transpose(0,1,2,3,4,6,5)
        SzB_imp = np.tensordot(SzB_psi,psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SzB_imp = SzB_imp.reshape(dim**2, dim**2, dim**2, dim**2)
        SxB_psi = np.tensordot(psi, Sx, ([5],[1])).transpose(0,1,2,3,4,6,5)
        SxB_imp = np.tensordot(SxB_psi,psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SxB_imp = SxB_imp.reshape(dim**2, dim**2, dim**2, dim**2)

        # Magnetization on C site
        SzC_psi = np.tensordot(psi, Sz, ([6],[1]))
        SzC_imp = np.tensordot(SzC_psi,psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SzC_imp = SzC_imp.reshape(dim**2, dim**2, dim**2, dim**2)
        SxC_psi = np.tensordot(psi, Sx, ([6],[1]))
        SxC_imp = np.tensordot(SxC_psi,psi,([4,5,6],[4,5,6])).transpose(0,4,1,5,2,6,3,7)
        SxC_imp = SxC_imp.reshape(dim**2, dim**2, dim**2, dim**2)

        return Ta_imp, SxA_imp, SzA_imp, SxB_imp, SzB_imp, SxC_imp, SzC_imp

    E_imp, SxA_imp, SzA_imp, SxB_imp, SzB_imp, SxC_imp, SzC_imp = Calc_impurity_tensor(ta)

    Env = np.transpose(
        np.tensordot(
            np.tensordot(
                T1, np.tensordot(
                    C1, T4, ([0], [2])
                ), ([0], [0])
            ), np.tensordot(
                np.tensordot(
                    C2, T2, ([1], [0])
                ), np.tensordot(
                    C3, np.tensordot(
                        T3, C4, ([2], [0])
                    ), ([1], [0])
                ), ([2], [0])
            ), ([1, 2], [0, 3])
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
    #print(s[0:4])

    return P, P_til

def CTM_corner(C1,C2,C3,C4,T11,T12,T21,T22,T31,T32,T41,T42,A1,A2,A3,A4):

    x1=(x+1)%Lx ; x2=(x+2)%Lx ; x3=(x+3)%Lx
    y1=(y+1)%Ly ; y2=(y+2)%Ly ; y3=(y+3)%Ly


    ## それぞれの四隅にまとめる
    
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
    if len(sys.argv) < 2:  D = 2
    else:  D = int(sys.argv[1])
    if len(sys.argv) < 3:  Hz = 0.0
    else:  Hz = float(sys.argv[2])
    if len(sys.argv) < 4:  dt = 0.1
    else:  dt = float(sys.argv[3])
    if len(sys.argv) < 5:  chi = D**2
    else:  chi = int(sys.argv[4])
    if len(sys.argv) < 6:  maxstepTEBD =4000
    else:  maxstepTEBD = int(sys.argv[5])
    if len(sys.argv) < 7:  maxstepCTM = 10
    else:  maxstepCTM = int(sys.argv[6])

    # open the text file
    #name = 'D'+str(D)+'-Hz'+str(Hz)
    #f = open(name+'.txt','w')

    D_cut = D
    J = 1.0
    spin = 0.5
    d_spin = int(2*spin + 1 )
    tau = dt

    # criterion for convergence
    eps_TEBD = 10**(-10);  eps_CTM = 10**(-10)
    
    # Heisenberg
    H, Ham = Hamiltonian_Heisen_In_Trian(J,Hz,spin)
    U = expm(-dt*H).reshape(d_spin, d_spin, d_spin, d_spin, d_spin, d_spin)


    # intiail iPESS
    A1, A2, A3, B1, B2, B3, C1, C2, C3,\
    R1_up, R2_up, R3_up, R1_low, R2_low, R3_low,\
    l_A1_up, l_A2_up, l_A3_up,\
    l_A1_low, l_A2_low, l_A3_low,\
    l_B1_up, l_B2_up, l_B3_up,\
    l_B1_low, l_B2_low, l_B3_low,\
    l_C1_up, l_C2_up, l_C3_up,\
    l_C1_low, l_C2_low, l_C3_low, =initial_iPESS(D,d_spin)


        
    for i in range(maxstepTEBD):

        U = expm(-tau*H).reshape(d_spin, d_spin,d_spin, d_spin, d_spin, d_spin)


        A1, B1, C1, R1_low, l_A1_low, l_B1_low, l_C1_low = \
        SimpleUpdate_down(A1,B1,C1,R1_low,l_A1_up,l_B1_up,l_C1_up,U)

        
        A2, B2, C2, R2_low, l_A2_low, l_B2_low, l_C2_low = \
        SimpleUpdate_down(A2,B2,C2,R2_low,l_A2_up,l_B2_up,l_C2_up,U)

        A3, B3, C3, R3_low, l_A3_low, l_B3_low, l_C3_low = \
        SimpleUpdate_down(A3,B3,C3,R3_low,l_A3_up,l_B3_up,l_C3_up,U) 
        
        
        B1, C2, A3, R2_up, l_B1_up, l_C2_up, l_A3_up = \
        SimpleUpdate_up(B1,C2,A3,R2_up,l_B1_low,l_C2_low,l_A3_low,U)

        B2, C3, A1, R3_up, l_B2_up, l_C3_up, l_A1_up = \
        SimpleUpdate_up(B2,C3,A1,R3_up,l_B2_low,l_C3_low,l_A1_low,U)

        B3, C1, A2, R1_up, l_B3_up, l_C1_up, l_A2_up = \
        SimpleUpdate_up(B3,C1,A2,R1_up,l_B3_low,l_C1_low,l_A2_low,U)

        ##  Calculate Energy
        E1_down = Energy_Triangle(A1,B1,C1,R1_low, l_A1_up, l_B1_up, l_C1_up, H, Ham)
        E1_up   = Energy_Triangle(B1.transpose(2,1,0), C2.transpose(2,1,0), A3.transpose(2,1,0), R2_up, l_B1_low, l_C2_low, l_A3_low, H, Ham)

        E2_down = Energy_Triangle(A2,B2,C2,R2_low, l_A2_up, l_B2_up, l_C2_up, H, Ham)
        E2_up   = Energy_Triangle(B2.transpose(2,1,0), C3.transpose(2,1,0), A1.transpose(2,1,0), R3_up, l_B2_low, l_C3_low, l_A1_low, H, Ham)

        E3_down = Energy_Triangle(A3,B3,C3,R3_low, l_A3_up, l_B3_up, l_C3_up, H, Ham)
        E3_up   = Energy_Triangle(B3.transpose(2,1,0), C1.transpose(2,1,0), A2.transpose(2,1,0), R1_up, l_B3_low, l_C1_low, l_A2_low, H, Ham)

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
        """
        if i%100 ==0:
            
            print(i, Mz/spin)
            print(i, E1, E1_down, E1_up )
            print(i, E2, E2_down, E2_up )
            print(i, E3, E3_down, E3_up ,"\n")
            print("A1 site: ",mxa1, mza1, np.sqrt(mxa1**2 + mza1**2))
            print("A2 site: ",mxa2, mza2, np.sqrt(mxa2**2 + mza2**2))
            print("A3 site: ",mxa3, mza3, np.sqrt(mxa3**2 + mza3**2))

            print("B1 site: ",mxb1, mzb1, np.sqrt(mxb1**2 + mzb1**2))
            print("B2 site: ",mxb2, mzb2, np.sqrt(mxb2**2 + mzb2**2))
            print("B3 site: ",mxb3, mzb3, np.sqrt(mxb3**2 + mzb3**2))

            print("C1 site: ",mxc1, mzc1, np.sqrt(mxc1**2 + mzc1**2))
            print("C2 site: ",mxc2, mzc2, np.sqrt(mxc2**2 + mzc2**2))
            print("C3 site: ",mxc3, mzc3, np.sqrt(mxc3**2 + mzc3**2))
        """
            

         
        if (i + 1)%1000 ==0: 
            tau /= 10
            #D_cut += 1
        
    #print("A: ",mxa1, mza1, np.sqrt(mxa1**2 + mza1**2))
    #print("B: ",mxb1, mzb1, np.sqrt(mxb1**2 + mzb1**2))
    #print("C: ",mxc1, mzc1, np.sqrt(mxc1**2 + mzc1**2), "\n")


    ta1 = Calcu_Unit_down(A1,B1,C1, R1_up, R1_low)
    ta2 = Calcu_Unit_down(A2,B2,C2, R2_up, R2_low)
    ta3 = Calcu_Unit_down(A3,B3,C3, R3_up, R3_low)



    ####### CTMRG ###############


    Lx=2; Ly=2
    SU_ten =[ [ta1,ta2], [ta2,ta3] ]
    CTMs = [[0 for y in range(Ly)]  for x in range(Lx)  ]


    ## SUで得られた初期テンソルをクラスのリストに代入する
    for x, y in product(range(Lx), range(Ly) ):
        tensor = Tensors_CTM(SU_ten[x][y])
        CTMs[x][y] = tensor

    for i in range(maxstepCTM):
        Mz = 0.0
        CTMRG(CTMs, Lx, Ly)
        ## 期待値を求める ##
        for x, y in product(range(Lx), range(Ly) ):
            x1 = (x+1)%Lx ; x2 = (x+2)%Lx 
            y1 = (y+1)%Ly ; y2 = (y+2)%Ly

            E_down, mxa, mza, mxb, mzb, mxc, mzc= ComputeQuantities_down_triangle(\
                SU_ten[x1][y1], CTMs[x1][y1].Ta,\
                CTMs[x][y].C1, CTMs[x2][y].C2, CTMs[x2][y2].C3, CTMs[x][y2].C4,\
                CTMs[x1][y].T1, CTMs[x2][y1].T2, CTMs[x1][y2].T3, CTMs[x][y1].T4,H)

            Mz += mza + mzb + mzc

            #print(i, x1,y1, E_down)
            #print("A site: ",mxa, mza, np.sqrt(mxa**2 + mza**2))
            #print("B site: ",mxb, mzb, np.sqrt(mxb**2 + mzb**2))
            #print("C site: ",mxc, mzc, np.sqrt(mxc**2 + mzc**2),"\n")
            """
            theta = (mxa*mxb+mza*mzb)/(np.sqrt(mxa**2 + mza**2) * np.sqrt(mxb**2 + mzb**2))
            theta=np.arccos(theta)
            print(np.rad2deg(theta))

            theta = (mxa*mxc+mza*mzc)/(np.sqrt(mxa**2 + mza**2) * np.sqrt(mxc**2 + mzc**2))
            theta=np.arccos(theta)
            print(np.rad2deg(theta))

            theta = (mxc*mxb+mzc*mzb)/(np.sqrt(mxc**2 + mzc**2) * np.sqrt(mxb**2 + mzb**2))
            theta=np.arccos(theta)
            print(np.rad2deg(theta))
            """
        #print(i, Mz_tot/12./spin) 

    """        
    print("A1 site: ",mxa1, mza1, np.sqrt(mxa1**2 + mza1**2))
    print("A2 site: ",mxa2, mza2, np.sqrt(mxa2**2 + mza2**2))
    print("A3 site: ",mxa3, mza3, np.sqrt(mxa3**2 + mza3**2))

    print("B1 site: ",mxb1, mzb1, np.sqrt(mxb1**2 + mzb1**2))
    print("B2 site: ",mxb2, mzb2, np.sqrt(mxb2**2 + mzb2**2))
    print("B3 site: ",mxb3, mzb3, np.sqrt(mxb3**2 + mzb3**2))

    print("C1 site: ",mxc1, mzc1, np.sqrt(mxc1**2 + mzc1**2))
    print("C2 site: ",mxc2, mzc2, np.sqrt(mxc2**2 + mzc2**2))
    print("C3 site: ",mxc3, mzc3, np.sqrt(mxc3**2 + mzc3**2),"\n")
        
    Mz =(mza1 + mza2 + mza3 + mzb1 + mzb2 + mzb3 + mzc1 + mzc2 + mzc3)/9. 
    """
    print(Hz, Mz/spin/12.)
    
    

        
        
        
            
            
          

























