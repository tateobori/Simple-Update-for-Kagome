# -*- coding: utf-8 -*-

import scipy.linalg as spl
import numpy as np

def psvd(mat, rank):
    m,n = mat.shape
    rank = min(rank,m,n)
    try:
        u, s, vt = spl.svd(mat, full_matrices=False)
        #print("berore",s,"\n")
    except np.linalg.linalg.LinAlgError:
        print('svd error:',mat)
        u, s, vt = spl.svd(mat, full_matrices=False,lapack_driver='gesvd')
    return u[:,:rank], s[:rank], vt[:rank,:]

def peigh(mat, rank):
    m,n = mat.shape
    rank = min(rank,m,n)
    try:
        u, s, vt = spl.svd(mat, full_matrices=False)
        #print("berore",s,"\n")
    except np.linalg.linalg.LinAlgError:
        print('eigh error:',mat)
        u, s = spl.eigh(mat)
    return u[:,:rank], s[:rank]

def tensor_svd(a,axes0,axes1,rank):
    shape = np.array(a.shape)
    shape_row = [ shape[i] for i in axes0 ]
    shape_col = [ shape[i] for i in axes1 ]
    n_row = np.prod(shape_row)
    n_col = np.prod(shape_col)
    mat = np.reshape(np.transpose(a,axes0+axes1), (n_row,n_col))
    u, s, vt = psvd(mat, rank)
    return u.reshape(shape_row+[len(s)]), s, vt.reshape([len(s)]+shape_col)

def tensor_eigh(a,axes0,axes1,rank):
    shape = np.array(a.shape)
    shape_row = [ shape[i] for i in axes0 ]
    shape_col = [ shape[i] for i in axes1 ]
    n_row = np.prod(shape_row)
    n_col = np.prod(shape_col)
    mat = np.reshape(np.transpose(a,axes0+axes1), (n_row,n_col))
    u, s = peigh(mat, rank)
    return u.reshape(shape_row+[len(s)]), s

def symmetric_svd(mat,rank):

    m,n = mat.shape
    rank = min(rank,m,n)
    try:
        u, s, vt = spl.svd(mat, full_matrices=False)
        s=np.sqrt(s)
        u=u*s[None,:]; vt = vt*s[:,None]
    except np.linalg.linalg.LinAlgError:
        print('svd error:',mat)
        u, s, vt = spl.svd(mat, full_matrices=False,lapack_driver='gesvd')
        s=np.sqrt(s)
        u=u*s[None,:]; vt = vt*s[:,None]
    return u[:,:rank], vt[:rank,:]

def pQR(mat, rank):
    m,n = mat.shape
    rank = min(rank,m,n)
    try:
        q,r = spl.qr(mat)
    except np.linalg.linalg.LinAlgError:
        print('qr decomposiition error:',mat)
        #q,r = spl.qr(mat, full_matrices=False,lapack_driver='gesvd')
    return q[:,:rank], r[:rank,:]
    
def tensor_QR(a,axes0,axes1,rank):
    shape = np.array(a.shape)
    shape_row = [ shape[i] for i in axes0 ]
    shape_col = [ shape[i] for i in axes1 ]
    n_row = np.prod(shape_row)
    n_col = np.prod(shape_col)
    mat = np.reshape(np.transpose(a,axes0+axes1), (n_row,n_col))
    q,r = pQR(mat, rank)
    return q.reshape(shape_row+[q.shape[1]]), r.reshape([q.shape[1]]+shape_col)
