#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:26:29 2022

@author: philippotn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr
from numba import njit
import sparse

#- Data LES
# simu = 'LBA'
# simu = 'AMMA'
# simu = 'AMA0W'
# simu = 'BOMEX'

simu = 'AMMA_end'
if simu == 'AMMA_end':
    path = 'C:/Users/Nathan/Documents/Stage_Toulouse/LES_AMMA/SIMU_LES/'
    lFiles = [path + 'AMMH3.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(280,361)]
    
if simu == 'AMMA':
    path = '/cnrm/tropics/user/couvreux/POUR_NATHAN/AMMA/SIMU_LES/'
#    lFiles = [path + 'AMMA2.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,13)]
#    lFiles = [path + 'AMMHF.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,73)]
#    lFiles = [path + 'AMMH2.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,73)]
    lFiles = [path + 'AMMH3.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,361)]
#    lFiles = [path + 'AMMH3.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,200)]

elif simu == 'AMA0W':
    path = '/cnrm/tropics/user/philippotn/LES_AMA0W/SIMU_LES/'
#    lFiles = [path + 'AMA0W.1.R6H5M.OUT.{:03d}.nc'.format(i) for i in range(70,73)]
    lFiles = [path + 'AMA0W.1.R6H1M.OUT.{:03d}.nc'.format(i) for i in range(1,361)]
    
elif simu == 'LBA':
    path = '/cnrm/tropics/user/philippotn/LES_LBA/SIMU_LES/'
    # lFiles = [path + 'LBA__.1.RFLES.OUT.{:03d}.nc'.format(i) for i in range(1,7)]
    # lFiles = [path + 'LBA__.1.10H1M.OUT.{:03d}.nc'.format(i) for i in range(200,361)]
    lFiles = [path + 'LBA_2.1.10H1M.OUT.{:03d}.nc'.format(i) for i in range(1,241)]
    
elif simu == 'BOMEX':
    path = '/cnrm/tropics/user/couvreux/POUR_AUDE/SIMU_MNH/bigLES/'
    # lFiles = [path + 'L25{:02d}.1.KUAB2.OUT.{:03d}.nc'.format(i//6+1,i%6+1) for i in range(120)]
    hours = np.arange(1,121,1)
    lFiles = [path + 'L25{:02d}.1.KUAB2.OUT.{:03d}.nc'.format(i//6+1,i%6+1) for i in hours-1]
    
# savePath = '/home/philippotn/Documents/'
savePath = 'C:/Users/Nathan/Documents/Code Python/Atmosphere/SkyView/'

f0 = xr.open_dataset(lFiles[-1])
z = np.array(f0.level)
nz = len(z)


#%%
@njit()
def interp_vertical(var,z,new_z):
    nt,nz,nx,ny = np.shape(var)
    new_var = np.zeros((nt,len(new_z),nx,ny),dtype=var.dtype)
    for iz in range(len(new_z)):
        io = np.argmin(np.abs(new_z[iz]-z))
        if new_z[iz]<z[io]:
            io-=1
        dz = z[io+1]-z[io]
        new_var[:,iz] = var[:,io]*(z[io+1]-new_z[iz])/dz + var[:,io+1]*(new_z[iz]-z[io])/dz

    return new_var

#%%
def emptyness(arrayCOO,end='\n'):
    print('Volume = {:02%}'.format(arrayCOO.nnz/arrayCOO.size),end=end)
    
dtype = np.float32
idx_dtype = np.int32
# new_z = np.linspace(0,12000,121,dtype=dtype)
new_z = np.linspace(0,20000,101,dtype=dtype)

for it,file in enumerate(lFiles):
    end = '--- '
    # print(simu +" {} ".format(it+1),end=end)
    print(simu+' '+file[-6:-3],end=end)
    f0 = xr.open_dataset(file)
    
    rcloud = np.array(f0.RCT,dtype=dtype) +np.array(f0.RIT,dtype=dtype)
    rprecip = np.array(f0.RRT,dtype=dtype)+np.array(f0.RST,dtype=dtype)+np.array(f0.RGT,dtype=dtype)
    rvapor = np.array(f0.RVT, dtype=dtype)

    thetav = np.array( f0.THT, dtype=dtype) * (1+ 1.61*rvapor)/(1+rvapor+rcloud+rprecip)
    
    print("Interpolation",end=end)
    new_rcloud = interp_vertical(rcloud ,z,new_z)
    new_rprecip = interp_vertical(rprecip,z,new_z)
    # new_w = interp_vertical(np.array(f0.WT,dtype=dtype)[:,1:],z[1:]/2+z[:-1]/2,new_z)
    new_thva = interp_vertical( thetav -np.median(thetav,axis=(2,3),keepdims=True) ,z,new_z )
    
    print("Threshold",end=end)
    new_rcloud[new_rcloud < 1e-6 ] = 0.
    new_rprecip[new_rprecip < 1e-4 ] = 0.
    # new_w[np.abs(new_w) < 2] = 0.
    # new_thva[np.abs(new_thva) < 1.5] = 0.
    new_thva[:,14:,:,:] = 0. # 11 for BOMEX # 14 for AMMA # 8 for LBA
    new_thva[new_thva > -0.3] = 0.
    
    print("Sparse",end=end)
    new_rcloud_sparse = sparse.COO.from_numpy(new_rcloud,idx_dtype=idx_dtype)
    new_rprecip_sparse = sparse.COO.from_numpy(new_rprecip,idx_dtype=idx_dtype)
    # new_w_sparse = sparse.COO.from_numpy(new_w,idx_dtype=idx_dtype)
    new_thva_sparse = sparse.COO.from_numpy(new_thva,idx_dtype=idx_dtype)
    if it==0:
        rcloud_sparse = new_rcloud_sparse
        rprecip_sparse = new_rprecip_sparse
        # w_sparse = new_w_sparse
        thva_sparse = new_thva_sparse
    else:
        rcloud_sparse = sparse.concatenate( (rcloud_sparse, new_rcloud_sparse ),axis=0)
        rprecip_sparse = sparse.concatenate( (rprecip_sparse, new_rprecip_sparse ),axis=0)
        # w_sparse = sparse.concatenate( (w_sparse, new_w_sparse ),axis=0)
        thva_sparse = sparse.concatenate( (thva_sparse, new_thva_sparse ),axis=0)
    print("Done",end=end)
    emptyness(new_rcloud_sparse,end=end)
    emptyness(new_rprecip_sparse,end=end)
    emptyness(new_thva_sparse)
    
sparse.save_npz(savePath+'rcloud_sparse_'+simu ,rcloud_sparse)    
sparse.save_npz(savePath+'rprecip_sparse_'+simu ,rprecip_sparse)
# sparse.save_npz(savePath+'w_sparse_'+simu ,w_sparse)
sparse.save_npz(savePath+'thva_sparse_'+simu ,thva_sparse)

# sparse.save_npz(savePath+'rcloud_sparse2_'+simu ,rcloud_sparse)    
# sparse.save_npz(savePath+'rprecip_sparse2_'+simu ,rprecip_sparse)
# # sparse.save_npz(savePath+'w_sparse2_'+simu ,w_sparse)
# sparse.save_npz(savePath+'thva_sparse2_'+simu ,thva_sparse)

# sparse.save_npz(savePath+'rcloud_sparse3_'+simu ,rcloud_sparse)    
# sparse.save_npz(savePath+'rprecip_sparse3_'+simu ,rprecip_sparse)
# # sparse.save_npz(savePath+'w_sparse3_'+simu ,w_sparse)
# sparse.save_npz(savePath+'thva_sparse3_'+simu ,thva_sparse)
#%%

emptyness(rcloud_sparse)
emptyness(rprecip_sparse)
# emptyness(w_sparse)
emptyness(thva_sparse)

#%%
def precise_emptyness(file):
    f0 = xr.open_dataset(file)
    
    N = np.size(np.array(f0.RCT)) /100
    rc = np.array(f0.RCT)
    ri = np.array(f0.RIT)
    rr = np.array(f0.RRT)
    rs = np.array(f0.RST)
    rg = np.array(f0.RGT)
    
    print('fraction liq= ',np.sum(rc>10**-6)/N)
    print('fraction ice = ',np.sum(ri>10**-6)/N)
    print('fraction cloud = ',np.sum(rc+ri>10**-6)/N,end=' \n \n')
    
    print('fraction rain = ',np.sum(rr>10**-4)/N)
    print('fraction snow = ',np.sum(rs>10**-4)/N)
    print('fraction graupel = ',np.sum(rg>10**-4)/N)
    print('fraction precip = ',np.sum(rr+rs+rg>10**-4 )/N)
    print('fraction precip hors nuages= ',np.sum(np.logical_and(rr+rs+rg>10**-4 , rc+ri<10**-6))/N,end=' \n \n')
        
precise_emptyness(lFiles[-1])
#%%

# #%%
# rcloud_sparse1 = sparse.load_npz(savePath+'rcloud_sparse2_'+simu+'.npz' )    
# rprecip_sparse1 = sparse.load_npz(savePath+'rprecip_sparse2_'+simu+'.npz' )
# # w_sparse1 = sparse.load_npz(savePath+'w_sparse2_'+simu+'.npz' )
# thva_sparse1 = sparse.load_npz(savePath+'thva_sparse2_'+simu+'.npz' )

# rcloud_sparse1 = sparse.load_npz(savePath+'rcloud_sparse_'+simu+'.npz' )    
# rprecip_sparse1 = sparse.load_npz(savePath+'rprecip_sparse_'+simu+'.npz' )
# w_sparse1 = sparse.load_npz(savePath+'w_sparse_'+simu+'.npz' )
# thva_sparse1 = sparse.load_npz(savePath+'thva_sparse_'+simu+'.npz' )

# rcloud_sparse2 = sparse.load_npz(savePath+'rcloud_sparse2_'+simu+'.npz' ) 
# rprecip_sparse2 = sparse.load_npz(savePath+'rprecip_sparse2_'+simu+'.npz' )
# # w_sparse2 = sparse.load_npz(savePath+'w_sparse2_'+simu+'.npz' )
# thva_sparse2 = sparse.load_npz(savePath+'thva_sparse2_'+simu+'.npz' )
# #%%
# rcloud_sparse2 = sparse.load_npz(savePath+'rcloud_sparse3_'+simu+'.npz' ) 
# rprecip_sparse2 = sparse.load_npz(savePath+'rprecip_sparse3_'+simu+'.npz' )
# # w_sparse2 = sparse.load_npz(savePath+'w_sparse3_'+simu+'.npz' )
# thva_sparse2 = sparse.load_npz(savePath+'thva_sparse3_'+simu+'.npz' )

# rcloud_sparse = sparse.concatenate( (rcloud_sparse1, rcloud_sparse2 ),axis=0)
# rprecip_sparse = sparse.concatenate( (rprecip_sparse1, rprecip_sparse2 ),axis=0)
# # w_sparse = sparse.concatenate( (w_sparse1, w_sparse2 ),axis=0)
# thva_sparse = sparse.concatenate( (thva_sparse1, thva_sparse2 ),axis=0)