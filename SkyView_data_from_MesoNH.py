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

simu = 'LBA'
z_max = 16001
orog = True

if simu == 'AMMA':
    path = '/cnrm/tropics/user/couvreux/POUR_NATHAN/AMMA/SIMU_LES/'
    lFiles = [path + 'AMMH3.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,361)]

elif simu == 'AMA0W':
    path = '/cnrm/tropics/user/philippotn/LES_AMA0W/SIMU_LES/'
    lFiles = [path + 'AMA0W.1.R6H1M.OUT.{:03d}.nc'.format(i) for i in range(1,361)]
    
elif simu == 'LBA':
    path = '/cnrm/tropics/user/philippotn/LES_LBA/SIMU_LES/'
    lFiles = [path + 'LBA__.1.10H1M.OUT.{:03d}.nc'.format(i) for i in range(200,361)] + [path + 'LBA_2.1.10H1M.OUT.{:03d}.nc'.format(i) for i in range(1,241)]
    
elif simu == 'BOMEX':
    path = '/cnrm/tropics/user/couvreux/POUR_AUDE/SIMU_MNH/bigLES/'
    hours = np.arange(1,121,1)
    lFiles = [path + 'L25{:02d}.1.KUAB2.OUT.{:03d}.nc'.format(i//6+1,i%6+1) for i in hours-1]
elif simu == 'AMOPL':
    dataPath = '/cnrm/tropics/user/philippotn/LES_AMOPL/SIMU_LES/'
    lFiles = [dataPath + 'AMOPL.1.200m1.OUT.{:03d}.nc'.format(i) for i in range(1,241,1)] + [dataPath + 'AMOPL.1.200m2.OUT.{:03d}.nc'.format(i) for i in range(1,722,1)]

    
savePath = '/cnrm/tropics/user/philippotn/SkyView/'

f0 = xr.open_dataset(lFiles[-1])
x = np.array(f0.ni)
dx = x[2]-x[1]
z = np.array(f0.level)
nz = len(z)
dtype = np.float32
idx_dtype = np.int32

new_z = np.arange(0.,z_max,dx,dtype=dtype)

if orog:
    ZS = xr.open_dataset(dataPath+simu+'_init_R'+str(round(dx))+'m_pgd.nc')['ZS'].data
    @njit()
    def interp_vertical(var,Zm,Z,nan=False): # var has to be np.float32
        # var (3D) =  variable defined on Zm levels with Gal-Chen and Somerville terrain-following coordinates
        # Z (1D) =  altitude levels on which new_var in interpolated
        # Zm (1D) = terrain-following model levels
        # ZS (2D) = surface altitude
        nt,_,nx,ny = np.shape(var)
        nz, = np.shape(Z)
        ZTOP = Zm[-1]
        if nan
        new_var = np.zeros((nt,nz,nx,ny),dtype=var.dtype)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    zs = ZS[i,j]
                    l = 0
                    if Z[k]<zs:
                        continue
                    else:
                        zm = ZTOP * (Z[k]-zs) / (ZTOP-zs)
                        while Zm[l+1]<zm:
                            l+=1
                        dZ = Zm[l+1]-Zm[l]
                        new_var[:,k,i,j] = ( var[:,l,i,j]*(Zm[l+1]-zm) + var[:,l+1,i,j]*(zm-Zm[l]) )/dZ
        return new_var
else:
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

def emptyness(arrayCOO,end='\n'):
    print('Volume = {:02%}'.format(arrayCOO.nnz/arrayCOO.size),end=end)

for it,file in enumerate(lFiles):
    end = '--- '
    # print(simu +" {} ".format(it+1),end=end)
    print(simu+' '+file[-6:-3],end=end)
    f0 = xr.open_dataset(file)
    
    rcloud = np.array(f0.RCT,dtype=dtype) +np.array(f0.RIT,dtype=dtype)
    rprecip = np.array(f0.RRT,dtype=dtype)+np.array(f0.RST,dtype=dtype)+np.array(f0.RGT,dtype=dtype)
    rvapor = np.array(f0.RVT, dtype=dtype)
    rtracer = np.array(f0.SVT001, dtype=dtype)
    thetav = np.array( f0.THT, dtype=dtype) * (1+ 1.61*rvapor)/(1+rvapor+rcloud+rprecip)
    
    print("Interpolation",end=end)
    new_rcloud = interp_vertical(rcloud ,z,new_z)
    new_rprecip = interp_vertical(rprecip,z,new_z)
    new_rtracer = interp_vertical(rtracer,z,new_z)
    new_w = interp_vertical(np.array(f0.WT,dtype=dtype)[:,1:],z[1:]/2+z[:-1]/2,new_z)
    new_thva = interp_vertical( thetav,z,new_z )
    new_thva -= np.nanmean(thetav,axis=(2,3),keepdims=True)
    
    print("Threshold",end=end)
    new_rcloud[new_rcloud < 1e-6 ] = np.nan
    new_rprecip[new_rprecip < 1e-4 ] = np.nan
    new_rtracer[new_rtracer < 1. ] = np.nan
    new_w[np.abs(new_w) < 2.] = np.nan
    # new_thva[np.abs(new_thva) < 2] = np.nan
    new_thva[:,20:,:,:] = np.nan # 11 for BOMEX # 14 for AMMA # 8 for LBA # 20 for AMOPL
    new_thva[new_thva > -1] = np.nan
    
    print("Sparse",end=end)
    new_rcloud_sparse = sparse.COO.from_numpy(new_rcloud,idx_dtype=idx_dtype,fill_value=np.nan)
    new_rprecip_sparse = sparse.COO.from_numpy(new_rprecip,idx_dtype=idx_dtype,fill_value=np.nan)
    new_rtracer_sparse = sparse.COO.from_numpy(new_rtracer,idx_dtype=idx_dtype,fill_value=np.nan)
    new_w_sparse = sparse.COO.from_numpy(new_w,idx_dtype=idx_dtype,fill_value=np.nan)
    new_thva_sparse = sparse.COO.from_numpy(new_thva,idx_dtype=idx_dtype,fill_value=np.nan)
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
    print("Done",end=end)
    emptyness(new_rcloud_sparse,end=end)
    emptyness(new_rprecip_sparse,end=end)
    emptyness(new_rtracer_sparse,end=end)
    emptyness(new_thva_sparse)
    
sparse.save_npz(savePath+'rcloud_sparse_'+simu ,rcloud_sparse)  
sparse.save_npz(savePath+'rprecip_sparse_'+simu ,rprecip_sparse)
sparse.save_npz(savePath+'rtracer_sparse_'+simu ,rtracer_sparse)
sparse.save_npz(savePath+'w_sparse_'+simu ,w_sparse)
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
emptyness(rtracer_sparse)
emptyness(w_sparse)
emptyness(thva_sparse)

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