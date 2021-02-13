# -*- coding: utf-8 -*-

import sys
import time
from scipy.linalg import expm
import scipy.linalg as spl
import numpy as np
from msvd import tensor_svd
from msvd import tensor_eigh
import CTMRG2x2 as CTMRG

################################################################
def spin_operators(S):
    """" Returns the spin operators  """    
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

def Hamiltonian_Ising_In_Trian(J,hx):

    Sx, Sy, Sz = spin_operators(0.5)
    I =np.eye(2,2)
    H_BC = np.kron(I, np.kron(Sz,Sz))
    H_AB = np.kron(np.kron(Sz,Sz), I)
    H_CA = np.kron(np.kron(Sz,I), Sz)

    Ham = J*np.kron(Sz,Sz) - 0.25 * hx *( np.kron(Sx,I) + np.kron(I,Sx) )
    H =  J*(H_AB + H_BC + H_CA) -0.5*hx*( np.kron(I, np.kron(I,Sx)) + np.kron(I, np.kron(Sx,I)) +  np.kron(Sx, np.kron(I,I)))



    return H, Ham.reshape(2,2,2,2)
###########################################################################
def initial_iPESS(Dx):

    ## random
    
    
    A = np.random.random((Dx, d_spin, Dx))# +1.0j
    B = np.random.random((Dx, d_spin, Dx))# +1.0j 
    C = np.random.random((Dx, d_spin, Dx))# + 1.0j
    R = np.random.random((Dx,Dx,Dx)) 
    #R = np.ones((Dx,Dx,Dx))
    #R[0,0,0]=1.
    
    ## classical uud state only can be applied to Dx=1
    
    """
    A = np.zeros((Dx,2,Dx))# + 1.0j
    B = np.zeros((Dx,2,Dx))# + 1.0j 
    C = np.zeros((Dx,2,Dx))# + 1.0j
    R  = np.zeros((Dx,Dx,Dx))# + 1.0j

    A[0,0,0]=1. ; B[0,1,0]=1.
    C[0,0,0]=1. 
    R[0,0,0]=1.
    """
    
    
    

    # vector lu, lr, ld, ll
    l = np.ones(A.shape[0], dtype=float)
    #for i in np.arange(len(l)):    l[i] /= 10**i
    #l /= np.sqrt(np.dot(l,l))
    
    return A, B, C, R, R, l, l, l, l, l, l

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

    return np.real(E), E_AB, E_BC, E_CA

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



##  main function
if __name__=="__main__":
    # obtain the arguments
    if len(sys.argv) < 2:  D = 2
    else:  D = int(sys.argv[1])
    if len(sys.argv) < 3:  Hz = 0.0
    else:  Hz = float(sys.argv[2])
    if len(sys.argv) < 4:  dt = 0.01
    else:  dt = float(sys.argv[3])
    if len(sys.argv) < 5:  chi = D**2
    else:  chi = int(sys.argv[4])
    if len(sys.argv) < 6:  maxstepTEBD = 12000
    else:  maxstepTEBD = int(sys.argv[5])
    if len(sys.argv) < 7:  maxstepCTM = 5
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
    
    A, B, C, R_up, R_low, \
    la_up, lb_up, lc_up, la_low, lb_low, lc_low = initial_iPESS(D)


    for i in range(maxstepTEBD):
      
        #tau = 10**(np.log10(dt)-4.*i/maxstepTEBD)
        #U = expm(-tau*H).reshape(2,2,2,2,2,2)

        U = expm(-tau*H).reshape(d_spin, d_spin, d_spin, d_spin, d_spin, d_spin)

        A, B, C, R_low, la_low, lb_low, lc_low = \
        SimpleUpdate_down(A, B, C, R_low, la_up, lb_up, lc_up, U)

        B, C, A, R_up, lb_up, lc_up, la_up = \
        SimpleUpdate_down(B.transpose(2,1,0), C.transpose(2,1,0), A.transpose(2,1,0), R_up, lb_low, lc_low, la_low, U)

        B = B.transpose(2,1,0) ; C = C.transpose(2,1,0); A = A.transpose(2,1,0)

        E_down, E_AB, E_BC, E_CA = Energy_Triangle(A,B,C,R_low, la_up, lb_up, lc_up, H, Ham)
        E_up, E_BC1, E_CA1, E_AB1   = Energy_Triangle(B.transpose(2,1,0),C.transpose(2,1,0),A.transpose(2,1,0), R_up, lb_low, lc_low, la_low, H, Ham)


        mxa, mya, mza = Magnetization(A, la_up, la_low)
        mxb, myb, mzb = Magnetization(B, lb_up, lb_low)
        mxc, myc, mzc = Magnetization(C, lc_up, lc_low)

        E = (E_up + E_down)/3. ## per site
        Mz =(mza + mzb + mzc)/3. 

        """
        if i%100==0:
            
            print(i, E, E_down, E_up)
            print(mxa, mza, np.sqrt(mxa**2 + mza**2))
            print(mxb, mzb, np.sqrt(mxb**2 + mzb**2))
            print(mxc, mzc, np.sqrt(mxc**2 + mzc**2), "\n")

            
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

        if (i + 1)%4000==0: 
            #D_cut +=1
            #tau = 0.01
            tau /=10

 
    # result
    print(E, Hz, Mz/spin)
    #f.write("{0:.8e}, {1:.8e}, {2:.8e},\n".format(E, Hz, Mz/spin))


        
        
        
            
            
          

























