#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:26:29 2022

@author: philippotn
"""
import sparse

simu = 'AMOPL'
savePath = '/cnrm/tropics/user/philippotn/SkyView/data_sparse/'

for it in range(961):
    print(it,end=' ')
    new_rcloud_sparse = sparse.load_npz(savePath+'rcloud_sparse_'+simu+'_'+str(it)+'.npz' )
    new_rprecip_sparse = sparse.load_npz(savePath+'rprecip_sparse_'+simu+'_'+str(it)+'.npz' )
    new_rtracer_sparse = sparse.load_npz(savePath+'rtracer_sparse_'+simu+'_'+str(it)+'.npz' )
    new_w_sparse = sparse.load_npz(savePath+'w_sparse_'+simu+'_'+str(it)+'.npz' )
    new_thva_sparse = sparse.load_npz(savePath+'thva_sparse_'+simu+'_'+str(it)+'.npz' )

    if it==0:
        rcloud_sparse = new_rcloud_sparse
        rprecip_sparse = new_rprecip_sparse
        rtracer_sparse = new_rtracer_sparse
        w_sparse = new_w_sparse
        thva_sparse = new_thva_sparse
    else:
        rcloud_sparse = sparse.concatenate( (rcloud_sparse, new_rcloud_sparse ),axis=0)
        rprecip_sparse = sparse.concatenate( (rprecip_sparse, new_rprecip_sparse ),axis=0)
        rtracer_sparse = sparse.concatenate( (rtracer_sparse, new_rtracer_sparse ),axis=0)
        w_sparse = sparse.concatenate( (w_sparse, new_w_sparse ),axis=0)
        thva_sparse = sparse.concatenate( (thva_sparse, new_thva_sparse ),axis=0)
    
sparse.save_npz(savePath+'rcloud_sparse_'+simu ,rcloud_sparse)  
sparse.save_npz(savePath+'rprecip_sparse_'+simu ,rprecip_sparse)
sparse.save_npz(savePath+'rtracer_sparse_'+simu ,rtracer_sparse)
sparse.save_npz(savePath+'w_sparse_'+simu ,w_sparse)
sparse.save_npz(savePath+'thva_sparse_'+simu ,thva_sparse)

def emptyness(arrayCOO,end='\n'):
    print('Volume = {:02%}'.format(arrayCOO.nnz/arrayCOO.size),end=end)
    
emptyness(rcloud_sparse,end=end)
emptyness(rprecip_sparse,end=end)
emptyness(rtracer_sparse,end=end)
emptyness(w_sparse,end=end)
emptyness(thva_sparse)