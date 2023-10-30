#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:26:29 2022

@author: philippotn
"""
import sparse
simu = 'AMOPL'
savePath = '/cnrm/tropics/user/philippotn/SkyView/data_sparse/'
def emptyness(arrayCOO,end='\n'):
    print('Volume = {:02%}'.format(arrayCOO.nnz/arrayCOO.size),end=end)
    
for var in ['rcloud','rprecip','rtracer','w','thva']:
    var_sparses = [ sparse.load_npz(savePath+var+'_sparse_'+simu+'_'+str(it)+'.npz' ) for it in range(961) ]
    var_sparse = sparse.concatenate( var_sparses ,axis=0)
    sparse.save_npz(savePath+var+'_sparse_'+simu ,var_sparse)  
    print(var,end=' ')
    emptyness(var_sparse)
