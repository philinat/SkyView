# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:53:32 2023

@author: Nathan
"""


import numpy as np
from numba import njit,prange
from math import cos,sin,tan,sqrt,exp,log10,pi,acos,atan2,erfc,ceil#,asin,erf,log,
import sparse
import matplotlib.image as mpimg
import pygame
from pygame import RLEACCEL

floatype = np.float32
intype = np.int16
complextype = np.complex64

sca_forward_ratio = 0.1
rho_water = 1000 # kg /m^3 # densité de l'eau condensée
rho0 = 1.292 # kg /m^3 # densité de l'air au niveau de la mer
Za = 8000 # m # epaisseur caracteristique de l'atmosphere pour le profil de densité
delta = 2.412e-32 # m^6 /mol # coefficient de Rayleigh pour l'air
Ma = 28.97e-3 # kg /mol # masse molaire air
Rt = 6350000 # m  # Rayon Terre
cloud_r_eff = 10e-6 # m # taille caractéristique des particules nuageuses
precip_r_eff = 5e-4 # m # taille caractéristique des particules precipitantes
Zaer = 1500
Baer = Zaer*3e-5
mag_range = 3.

lamda = np.array( [610.,535.,465.] ,dtype=floatype)
beta = delta*rho0*Za/Ma / (lamda*10**-9)**4 
sun_spec = np.array( [1.,1.,1.] ,dtype=floatype)

def day_of_year(day,month):
    return np.sum(np.array([31,28,31,30,31,30,31,31,30,31,30,31])[:month-1]) + day

#%%
# simu='BOMEX' ; surface_type = 'ocean' ; res = 100 ; dt = 1 ; time0 = 1 ; day = day_of_year(10,6) ; lon=-53 ; lat=13. ; timezone=-4
simu='AMMA_end' ; surface_type = 'image' ; res = 200 ; dt = 1/60 ; time0 = 16+40/60 ; day = day_of_year(10,6) ; lon=2 ; lat=13.7 ; timezone=1
# simu='AMMA' ; surface_type = 'image' ; res = 200 ; dt = 1/60 ; time0 = 12+dt ; day = day_of_year(10,6) ; lon=2 ; lat=13.7 ; timezone=1
# simu='AMA0W' ; surface_type = 'image' ; res = 200 ; dt = 1/60 ; time0 = 12+dt ; day = day_of_year(10,6) ; lon=2 ; lat=13.7 ; timezone=1
# simu='LBA' ; surface_type = 'image' ; res = 200 ; dt = 1/60 ; time0 = 9.5+dt ; day = day_of_year(23,2) ; lon=-60.1 ; lat=-3.15 ; timezone=-4
# simu='EXP01' ; surface_type = 'ocean' ; res = 100 ; dt = 1 ; time0 = 13 ; day = day_of_year(2,2) ; lon=-59 ; lat=12. ; timezone=-4

# simu='cheminee' ; surface_type = 'black' ; res = 0.1 ; dt = 4.6e-5 ; time0 = 13 ; day = day_of_year(16,1) ; lon=7 ; lat=46 ; timezone=1
# simu='myclouds' ; surface_type = 'black' ; res = 30 ; dt = 1/60 ; time0 = 12 ; day = day_of_year(10,6) ; lon=2 ; lat=13.7 ; timezone=1
    
sparse_names = ['rcloud']
sparse_names += ['rprecip']
# sparse_names += ['thva']
# sparse_names += ['w']

list_sparse = [sparse.load_npz(var+'_sparse_'+simu+'.npz')[:,:] for var in sparse_names ]

if surface_type == 'image':
    surface0 = np.array(np.flipud(mpimg.imread("surface_"+simu+".png")[:,:,:3]),dtype=floatype) 
elif surface_type == 'ocean':
    surface0 = -np.ones(list_sparse[0].shape[2:]+(3,),dtype=floatype)
elif surface_type == 'black':
    surface0 = 0.0001*np.ones(list_sparse[0].shape[2:]+(3,),dtype=floatype)
    # surface0[0,:,:] = 1 ;surface0[:,0,:] = 1 ; surface0[-1,:,:] = 1 ;surface0[:,-1,:] = 1 # bordure de domaine en blanc

nt,nz,nx,ny = list_sparse[0].shape
time = np.linspace(time0,time0+dt*(nt-1),nt)
list_ns = [sparse_.data.shape[0] for sparse_ in list_sparse ]

#%%# Data compatibility test
for v,sparse_ in enumerate(list_sparse):
    if (nt,nz,nx,ny) != list_sparse[v].shape:
        raise TypeError(sparse_names[v]+" sparse data doesn't have the same dimensions than "+sparse_names[0])
if (nx,ny) != surface0.shape[:2]:
    raise TypeError("Surface data doesn't have the same dimensions than "+sparse_names[0])

#%%
def sun_theta_phi(lon,lat,day,hour,timezone):
    gamma = 2*np.pi/365 * (day-1+(hour-12)/24)
    eqtime = 229.18*(0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma) - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma) )
    decl = 0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma) - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma) - 0.002697*np.cos(3*gamma) + 0.00148*np.sin (3*gamma)
    time_offset = eqtime + 4*lon - 60*timezone
    tst = hour*60 + time_offset
    ha = (tst/4 - 180)*np.pi/180
    lat = lat*np.pi/180
    theta_s = np.arccos( np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(ha))
    cos_phi_s = - (np.sin(lat)*np.cos(theta_s) - np.sin(decl))/ np.cos(lat)/np.sin(theta_s) 
    phi_s = np.sign(np.sin(ha))*np.arccos(cos_phi_s)
    return theta_s, np.pi/2- phi_s

les_theta_s , les_phi_s = sun_theta_phi(lon,lat,day,time,timezone)

#%%
@njit(parallel=True)
def surface_spectral_albedo(surface,mag_range):
    nx,ny,_ = surface.shape
    for i in prange(nx):
        for j in range(ny):
    # if i>=0 and i<=nx-1 and j>=0 and j<=ny-1:
            R,G,B = surface[i,j,0],surface[i,j,1],surface[i,j,2]
            M = max(R,G,B)
            m = min(R,G,B)
            C = M-m
            if C==0.:
                if M>0.:
                    for f in range(3):
                        surface[i,j,f] = 10**((surface[i,j,f]-1)*mag_range)
                        if surface[i,j,f]<0.:
                            surface[i,j,f]=0.
            else:
                if M==R:
                    H = ((G-B)/C)%6
                elif M==G:
                    H = (B-R)/C +2
                elif M==B:
                    H = (R-G)/C +4
                Luma0 = 0.2989*R + 0.5871*G + 0.1140*B
                Luma = 10**((Luma0-1)*mag_range)
                C*=Luma/Luma0
                X = C*(1-abs(H%2-1))
                if H<1:
                    R1,G1,B1 = C,X,0.
                elif H<2:
                    R1,G1,B1 = X,C,0.
                elif H<3:
                    R1,G1,B1 = 0.,C,X
                elif H<4:
                    R1,G1,B1 = 0.,X,C
                elif H<5:
                    R1,G1,B1 = X,0.,C
                elif H<6:
                    R1,G1,B1 = C,0.,X
                m = Luma - (0.2989*R1 + 0.5871*G1 + 0.1140*B1)
                surface[i,j,0],surface[i,j,1],surface[i,j,2] = R1+m,G1+m,B1+m
                for f in range(3):
                    if surface[i,j,f]<0.:
                        surface[i,j,f]=0.
                    elif surface[i,j,f]>1.:
                        surface[i,j,f]=1.
if surface_type == 'image':
    if simu=='LBA':
        surface_spectral_albedo(surface0,mag_range-1)
    else:
        surface_spectral_albedo(surface0,mag_range)
    # print('mean albedo =',2*pi*np.mean(surface0))
        

#%%
@njit()
def get_time_indexes(t_coords):
    i = 0
    time_indexes = np.zeros(nt+1,dtype=np.uint64)
    for it in range(nt):
        while i<len(t_coords) and t_coords[i]==it:
            i+=1
        time_indexes[it+1] = i
    return time_indexes

list_time_indexes = [ get_time_indexes(sparse_.coords[0]) for sparse_ in list_sparse ]
list_coords_sparse = [ np.array(sparse_.coords,dtype=intype) for sparse_ in list_sparse ]
# list_kext_sparse = [ np.zeros((ns),dtype=floatype) for ns in list_ns ]
list_kext_sparse = []
for v,var in enumerate(sparse_names):
    if var=='rcloud' or var=='rprecip':
        les_z = res*np.array(list_sparse[v].coords[1,:] ,dtype=floatype)
        rho_z = rho0*np.exp(-les_z/Za)
        if var=='rcloud': r_eff = cloud_r_eff
        if var=='rprecip': r_eff = precip_r_eff
        list_kext_sparse.append( np.array( sca_forward_ratio * 3/2 * res/r_eff * rho_z/rho_water * list_sparse[v].data ,dtype=floatype) )

    if var=='w' or var=='thva':
        list_kext_sparse.append( np.array( list_sparse[v].data ,dtype=floatype)) 
        
del list_sparse

#%% Mipmaps preparation
MMc = 2 # Mipmap coeficient
MMn = 0 # Mipmap number
n_min = min(nz,nx,ny)
while n_min>1:
    print(n_min)
    n_min = ceil(n_min/MMc)
    MMn += 1 

from numba.typed import List
sca = List([ np.zeros((nz,nx,ny,3),dtype=floatype) ])
k_ext = List([ np.zeros((nz,nx,ny),dtype=floatype) ])
# surface = [ np.copy(surface0) ]
surface = np.copy(surface0)

print("nz nx ny")
print(nz,nx,ny)
if MMn>0:
    nz_,nx_,ny_ = nz,nx,ny
    for i in range(MMn):
        nz_,nx_,ny_ = ceil(nz_/MMc),ceil(nx_/MMc),ceil(ny_/MMc)
        print(nz_,nx_,ny_)
        sca.append(np.zeros((nz_,nx_,ny_,3),dtype=floatype))
        k_ext.append(np.zeros((nz_,nx_,ny_),dtype=floatype))
        # surface.append(np.zeros((nx_,ny_,3),dtype=floatype))
        
@njit#(parallel=True)
def fill_kext_from_sparse(coords_sparse,kext_sparse,k_ext,t1,t2):
    for t in prange(t1,t2):
        for MM in range(MMn+1):
            k_ext[MM][coords_sparse[1,t]//(MMc**MM),coords_sparse[2,t]//(MMc**MM),coords_sparse[3,t]//(MMc**MM)] += kext_sparse[t]/(MMc**(MM*3))

 
# @njit(parallel=True)
# def fill_sca_and_kext_from_sparse_variable(coords_sparse,kext_sparse,sca,k_ext,t1,t2,intensity,opacity,tauG,tauRB):
#     for t in prange(t1,t2):
#         if kext_sparse[t] >0:
#             sca[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t],1] += kext_sparse[t]*intensity*exp(-kext_sparse[t]/tauG)
#             sca[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t],0] += kext_sparse[t]*intensity*exp(-kext_sparse[t]/tauRB)
#             k_ext[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t]] += kext_sparse[t]*intensity*opacity
#         else:
#             sca[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t],1] -= kext_sparse[t]*intensity*exp(kext_sparse[t]/tauG)
#             sca[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t],2] -= kext_sparse[t]*intensity*exp(kext_sparse[t]/tauRB)
#             k_ext[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t]] -= kext_sparse[t]*intensity*opacity

#%%
for MM in range(MMn+1):
    sca[MM][:] = 0.
    k_ext[MM][:] = 0.
it=0
show_sparse = [True,False,False,False]
for v,var in enumerate(sparse_names):
    if show_sparse[v]:
        if var=='rcloud' or var=='rprecip':
            fill_kext_from_sparse(list_coords_sparse[v],list_kext_sparse[v],k_ext,list_time_indexes[v][it],list_time_indexes[v][it+1])
#%%
import matplotlib.pyplot as plt
fig,ax = plt.subplots(ncols=MMn+1)
for MM in range(0,MMn+1,1):
    ax[MM].imshow(np.mean(k_ext[MM],axis=0),vmin=0,vmax=0.05)
    print(np.mean(k_ext[MM]))
#%% all functions
@njit#(parallel=True)
def solar_income_vertical_shadow_mipmaps(k_ext,sca,surface0,surface):
    nz,nx,ny = k_ext[0].shape
    for i in prange(nx):
        for j in range(ny):
            ray = 1.
            MM = 0#MMn
            iz = nz-1
            while iz>=0:
                # while MM<MMn and k_ext[MM][iz//MM,i//MM,j//MM]==0:
                # while MM<MMn and k_ext[MM+1][iz//(MMc**(MM+1)),i//(MMc**(MM+1)),j//(MMc**(MM+1))]==0:
                #     MM+=1
                # while MM>0 and k_ext[MM][iz//(MMc**MM),i//(MMc**MM),j//(MMc**MM)]>0:
                #     MM-=1
                if MM==0 and k_ext[0][iz,i,j]>0.:
                    frac = exp(-k_ext[0][iz,i,j])
                    for f in range(3):
                        # sca[0][iz,i,j,f] = (ray+0.01)*(1-frac)#/4/pi
                        for M in range(MMn+1):
                            sca[M][iz//(MMc**M),i//(MMc**M),j//(MMc**M),f] += (ray+0.01)*(1-frac)/(MMc**(M*3))
                    ray *= frac
                    iz -= 1
                else:
                    iz = iz//(MMc**MM) * MMc**MM - 1
            for f in range(3):
                surface[i,j,f] = surface0[i,j,f]*ray
                
solar_income_vertical_shadow_mipmaps(k_ext,sca,surface0,surface)
#%%
fig,ax = plt.subplots(ncols=3)
ax[0].imshow(np.mean(k_ext[0],axis=0),vmin=0,vmax=0.1)
ax[1].imshow(np.mean(sca[0],axis=(0,3)),vmin=0,vmax=0.015)
ax[2].imshow(surface)
#%%
@njit(parallel=True)
def solar_income_precise(res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,sca0):
    nz,nx,ny = k_ext.shape
    dz,dx,dy = -cos(theta_s), -sin(theta_s)*cos(phi_s), -sin(theta_s)*sin(phi_s)
    cos_s = cos(theta_s)
    
    if abs(dx)>abs(dz) and abs(dx)>=abs(dy):
        dx = dx/abs(dx) ; dz = dz/abs(dx) ; dy = dy/abs(dx)
        imax = nz-1 ; jmax = ny-1 ; steps = nx
    elif abs(dy)>abs(dz) and abs(dy)>=abs(dx):
        dy = dy/abs(dy) ; dz = dz/abs(dy) ; dx = dx/abs(dy)
        imax = nz-1 ; jmax = nx-1 ; steps = ny
    elif abs(dz)>=abs(dx) and abs(dz)>=abs(dy):
        dz = dz/abs(dz) ; dx = dx/abs(dz) ; dy = dy/abs(dz)
        imax = nx-1 ; jmax = ny-1 ; steps = nz
    D = sqrt(dz**2+dx**2+dy**2)
    
    for i in prange(imax+1):
        for j in range(jmax+1):
        # if i>=0 and i<=imax and j>=0 and j<=jmax:
            if dz==-1:
                z=nz-1 ; x=i+0.5 ; y=j+0.5
            elif dz==1:
                return
            elif dx==1:
                z=i+0.5 ; x=0 ; y=j+0.5
            elif dx==-1:
                z=i+0.5 ; x=nx-1 ; y=j+0.5
            elif dy==1:
                z=i+0.5 ; x=j+0.5 ; y=0
            elif dy==-1:
                z=i+0.5 ; x=j+0.5 ; y=ny-1
            
            
            z0 = z*res
            # if theta_s < acos(-sqrt(2*Rt*z0+z0**2)/Rt) + 0.009 :
            ray = np.zeros(3,dtype=floatype)
            for f in range(3):
                ray[f] = sun_spec[f] * exp( - beta[f]*exp(-z0/Za) * Chapman((Rt+z0)/Za,cos_s) - Baer*exp(-z0/Zaer) * Chapman((Rt+z0)/Zaer,cos_s))
            
            for step in range(steps):
                iz,ix,iy = int(z),int(x),int(y)
                if k_ext[iz,ix,iy]>0:
                    frac = exp( -D*k_ext[iz,ix,iy] )
                    for f in range(3):
                        sca0[iz,ix,iy,f] += (ray[f]+ 0.01)*(1-frac)
                        ray[f] *= frac
                z+=dz ; x+=dx ; y+=dy
                if not(z>=0. and z<nz and x>=0. and x<nx and y>=0. and y<ny):
                    if dz>=0:
                        break
                    else:
                        z=z%nz ; x=x%nx ; y=y%ny
                        z0 = z*res
                        for f in range(3):
                            ray[f] = sun_spec[f] * exp( - beta[f]*exp(-z0/Za) * Chapman((Rt+z0)/Za,cos_s) - Baer*exp(-z0/Zaer) * Chapman((Rt+z0)/Zaer,cos_s))


                
@njit(parallel=True)
def solar_income_vertical_shadow(sun_spec,k_ext,sca0,surface0,surface):
    nz,nx,ny = k_ext.shape
    for i in prange(nx):
        for j in range(ny):
            # ray = np.copy(sun_spec)
            ray = 1.
            for iz in range(nz-1,0,-1):
                if k_ext[iz,i,j]>0.:
                    frac = exp(-k_ext[iz,i,j])
                    for f in range(3):
                        sca0[iz,i,j,f] = (ray+0.01)*(1-frac)#/4/pi
                        # sca0[iz,i,j,f] = (ray[f]+0.01)*(1-frac)#/4/pi
                        # ray[f] *= frac
                    ray *= frac
            for f in range(3):
                surface[i,j,f] = surface0[i,j,f]*(ray)#/2/pi

@njit(parallel=True)
def surface_shadow(k_ext,theta_s,phi_s,surface0,surface):
    nz,nx,ny = k_ext.shape
    D = 1.#sqrt(3)
    dz,dx,dy = D*cos(theta_s), D*sin(theta_s)*cos(phi_s), D*sin(theta_s)*sin(phi_s)
    for i in prange(nx):
        for j in range(ny):
            shadow = 1.
            z=0. ; x=i+0.5 ; y=j+0.5
            while z>=0. and z<nz and x>=0. and x<nx and y>=0. and y<ny:
                iz,ix,iy = int(z),int(x),int(y)
                if k_ext[iz,ix,iy]>0:
                    shadow *= exp( -D*k_ext[iz,ix,iy] )
                z+=dz ; x+=dx ; y+=dy
            
            for f in range(3):
                surface[i,j,f] = surface0[i,j,f]*shadow
        
@njit
def projection_FOV(theta0,phi0,nh,nv,FOV,i,j):
    cos_theta0 = cos(theta0) ; sin_theta0 = sin(theta0)
    cos_phi0 = cos(phi0) ; sin_phi0 = sin(phi0)
    D = 2*tan(FOV/2)
    Di = (i/nh-0.5)*D
    Dj = (j-nv/2)*D/nh
    x = sin_theta0*cos_phi0 + Di*sin_phi0 - Dj*cos_theta0*cos_phi0
    y = sin_theta0*sin_phi0 - Di*cos_phi0 - Dj*cos_theta0*sin_phi0
    z = cos_theta0 + Dj*sin_theta0
    return acos(z/sqrt(x**2+y**2+z**2)) , atan2(y,x)

@njit
def mean_ray(a,b,c,dx,dy,dz,f000,f100,f010,f001,f011,f101,f110,f111):
    mean = 0.
    a_ = 1-a ; b_ = 1-b ; c_ = 1-c
    mean += f000 * ( a_*b_*c_ + ( -dx*b_*c_ -dy*a_*c_ -dz*a_*b_ )/2 + ( +dy*dz*a_ +dx*dz*b_ +dx*dy*c_ )/3 -dx*dy*dz/4 )
    mean += f100 * ( a *b_*c_ + ( +dx*b_*c_ -dy*a *c_ -dz*a *b_ )/2 + ( +dy*dz*a  -dx*dz*b_ -dx*dy*c_ )/3 +dx*dy*dz/4 )
    mean += f010 * ( a_*b *c_ + ( -dx*b *c_ +dy*a_*c_ -dz*a_*b  )/2 + ( -dy*dz*a_ +dx*dz*b  -dx*dy*c_ )/3 +dx*dy*dz/4 )
    mean += f001 * ( a_*b_*c  + ( -dx*b_*c  -dy*a_*c  +dz*a_*b_ )/2 + ( -dy*dz*a_ -dx*dz*b_ +dx*dy*c  )/3 +dx*dy*dz/4 )
    mean += f011 * ( a_*b *c  + ( -dx*b *c  +dy*a_*c  +dz*a_*b  )/2 + ( +dy*dz*a_ -dx*dz*b  -dx*dy*c  )/3 -dx*dy*dz/4 )
    mean += f101 * ( a *b_*c  + ( +dx*b_*c  -dy*a *c  +dz*a *b_ )/2 + ( -dy*dz*a  +dx*dz*b_ -dx*dy*c  )/3 -dx*dy*dz/4 )
    mean += f110 * ( a *b *c_ + ( +dx*b *c_ +dy*a *c_ -dz*a *b  )/2 + ( -dy*dz*a  -dx*dz*b  +dx*dy*c_ )/3 -dx*dy*dz/4 )
    mean += f111 * ( a *b *c  + ( +dx*b *c  +dy*a *c  +dz*a *b  )/2 + ( +dy*dz*a  +dx*dz*b  +dx*dy*c  )/3 +dx*dy*dz/4 )
    return mean

@njit
def Chapman(alpha,cos_s):
    gamma = cos_s*sqrt(alpha/2)
    if gamma < -4:
        return 1e10
    elif gamma < 8:
        return sqrt(pi*alpha/2)* exp(gamma**2) * erfc(gamma)
    else:
        return 1/cos_s
    
@njit
def mie_aerosol_phase_function(cosD):
    D = acos(cosD)
    power=1
    return (3*(1+cosD**2)+ exp((D-pi*3/4)*power)+exp(-(D-pi*2/3)*power*2.5)-2)/15.521057295924523
@njit
def sun_reflexion_phase_function(cosD):
    D = acos(cosD)
    power=3
    return (3*(1+cosD**2)+ exp((D-pi*3/4)*power)+exp(-(D-pi*2/3)*power*2.5)-2)/57101.15796667264

@njit
def sky_spec_round_earth_aerosols(z0,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,theta,phi,sky_spec):
    cos_s = cos(theta_s)
    cos_v = cos(theta)
    cosD = cos_v*cos_s + sin(theta)*sin(theta_s)*cos(phi-phi_s)
    n = 20
    z_max = Za*10
    z_max_aer = Zaer*5
    Deltat = sqrt((Rt+z0)**2*cos_v**2 + 2*Rt*(z_max-z0) + z_max**2-z0**2)
    t_min = max(0., -(Rt+z0)*cos_v - Deltat)
    t_max = -(Rt+z0)*cos_v + Deltat
    dt = (t_max-t_min)/n
    t = t_min
    while t<t_max:
        zt = -Rt + sqrt(Rt**2 + z0**2 + 2*Rt*z0 + 2*(Rt+z0)*t*cos_v + t**2 )
        if zt<z_max_aer:
            dtj = dt/int(Za/Zaer)
        else:
            dtj = dt
        alpha = sin(theta_s)*cos(phi-phi_s)* t*sin(theta)/(Rt+t*cos_v+z0)
        cos_se = cos(theta_s-alpha)
        if t == t_min:
            t += dtj
            dtj /= 2
            taut = np.zeros(3,dtype=floatype)
        else:
            t += dtj
        for f in range(3):
            taut[f] += dtj*( beta[f]/Za*exp(-zt/Za) + Baer/Zaer*exp(-zt/Zaer) )
            sun = sun_spec[f] * exp( - beta[f]*exp(-zt/Za) * Chapman((Rt+zt)/Za,cos_se) - Baer*exp(-zt/Zaer) * Chapman((Rt+zt)/Zaer,cos_se))
            sky_spec[f] += dtj * sun  * exp(-taut[f]) * ( 3/16/pi*(1+cosD**2)*beta[f]/Za*exp(-zt/Za) + mie_aerosol_phase_function(cosD)/4/pi*Baer/Zaer*exp(-zt/Zaer) )
    if cosD>0.9999:
        for f in range(3):
            sky_spec[f] += sun_spec[f] * exp(-taut[f])
            
@njit
def sky_spec_round_earth_aerosols_surface(z0,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,theta,phi,sky_spec,max_surface_sca):
    cos_s = cos(theta_s)
    cos_v = cos(theta)
    cosD = cos_v*cos_s + sin(theta)*sin(theta_s)*cos(phi-phi_s)
    n = 10
    z_max = Za*10
    z_aer = Zaer*3
    t_min = max(0., -(Rt+z0)*cos_v - sqrt((Rt+z0)**2*cos_v**2 + 2*Rt*(z_max-z0) + z_max**2-z0**2))
    t_aer = max(0., -(Rt+z0)*cos_v - sqrt((Rt+z0)**2*cos_v**2 + 2*Rt*(z_aer-z0) + z_aer**2-z0**2))
    t_surf = -(Rt+z0)*cos_v - sqrt((Rt+z0)**2*cos_v**2 - 2*Rt*z0 -z0**2)
    taut = np.zeros(3,dtype=floatype)
    if t_min<t_aer:
        dt = (t_aer-t_min)/n
        for it in range(n):
            t = t_min + (it+0.5)*dt
            zt = -Rt + sqrt(Rt**2 + z0**2 + 2*Rt*z0 + 2*(Rt+z0)*t*cos_v + t**2 )
            alpha = sin(theta_s)*cos(phi-phi_s)* t*sin(theta)/(Rt+t*cos_v+z0)
            cos_se = cos(theta_s-alpha)
            for f in range(3):
                taut[f] += dt*( beta[f]/Za*exp(-zt/Za) + Baer/Zaer*exp(-zt/Zaer) )
                sun = sun_spec[f] * exp( - beta[f]*exp(-zt/Za) * Chapman((Rt+zt)/Za,cos_se) - Baer*exp(-zt/Zaer) * Chapman((Rt+zt)/Zaer,cos_se))
                sky_spec[f] += dt * sun  * exp(-taut[f]) * ( 3/16/pi*(1+cosD**2)*beta[f]/Za*exp(-zt/Za) + mie_aerosol_phase_function(cosD)/4/pi*Baer/Zaer*exp(-zt/Zaer) )
    dt = (t_surf-t_aer)/n
    for it in range(n):
        t = t_aer + (it+0.5)*dt
        zt = -Rt + sqrt(Rt**2 + z0**2 + 2*Rt*z0 + 2*(Rt+z0)*t*cos_v + t**2 )
        alpha = sin(theta_s)*cos(phi-phi_s)* t*sin(theta)/(Rt+t*cos_v+z0)
        cos_se = cos(theta_s-alpha)
        for f in range(3):
            taut[f] += dt*( beta[f]/Za*exp(-zt/Za) + Baer/Zaer*exp(-zt/Zaer) )
            sun = sun_spec[f] * exp( - beta[f]*exp(-zt/Za) * Chapman((Rt+zt)/Za,cos_se) - Baer*exp(-zt/Zaer) * Chapman((Rt+zt)/Zaer,cos_se))
            sky_spec[f] += dt * sun  * exp(-taut[f]) * ( 3/16/pi*(1+cosD**2)*beta[f]/Za*exp(-zt/Za) + mie_aerosol_phase_function(cosD)/4/pi*Baer/Zaer*exp(-zt/Zaer) )
    if cos_se > 0:
        for f in range(3):
            sky_spec[f] += cos_se*max_surface_sca[f] * exp(-taut[f])
            
@njit
def sky_spec_round_earth_aerosols_ocean(z0,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,theta,phi,sky_spec,max_surface_sca):
    cos_s = cos(theta_s)
    cos_v = cos(theta)
    cosD = cos_v*cos_s + sin(theta)*sin(theta_s)*cos(phi-phi_s)
    n = 10
    z_max = Za*10
    z_aer = Zaer*3
    t_min = max(0., -(Rt+z0)*cos_v - sqrt((Rt+z0)**2*cos_v**2 + 2*Rt*(z_max-z0) + z_max**2-z0**2))
    t_aer = max(0., -(Rt+z0)*cos_v - sqrt((Rt+z0)**2*cos_v**2 + 2*Rt*(z_aer-z0) + z_aer**2-z0**2))
    t_surf = -(Rt+z0)*cos_v - sqrt((Rt+z0)**2*cos_v**2 - 2*Rt*z0 -z0**2)
    t_max = t_surf - 2*Za/cos_v
    taut = np.zeros(3,dtype=floatype)
    if t_min<t_aer:
        dt = (t_aer-t_min)/n
        for it in range(n):
            t = t_min + (it+0.5)*dt
            zt = -Rt + sqrt(Rt**2 + z0**2 + 2*Rt*z0 + 2*(Rt+z0)*t*cos_v + t**2 )
            alpha = sin(theta_s)*cos(phi-phi_s)* t*sin(theta)/(Rt+t*cos_v+z0)
            cos_se = cos(theta_s-alpha)
            for f in range(3):
                taut[f] += dt*( beta[f]/Za*exp(-zt/Za) + Baer/Zaer*exp(-zt/Zaer) )
                sun = sun_spec[f] * exp( - beta[f]*exp(-zt/Za) * Chapman((Rt+zt)/Za,cos_se) - Baer*exp(-zt/Zaer) * Chapman((Rt+zt)/Zaer,cos_se))
                sky_spec[f] += dt * sun  * exp(-taut[f]) * ( 3/16/pi*(1+cosD**2)*beta[f]/Za*exp(-zt/Za) + mie_aerosol_phase_function(cosD)/4/pi*Baer/Zaer*exp(-zt/Zaer) )
    dt = (t_surf-t_aer)/n
    for it in range(n):
        t = t_aer + (it+0.5)*dt
        zt = -Rt + sqrt(Rt**2 + z0**2 + 2*Rt*z0 + 2*(Rt+z0)*t*cos_v + t**2 )
        alpha = sin(theta_s)*cos(phi-phi_s)* t*sin(theta)/(Rt+t*cos_v+z0)
        cos_se = cos(theta_s-alpha)
        for f in range(3):
            taut[f] += dt*( beta[f]/Za*exp(-zt/Za) + Baer/Zaer*exp(-zt/Zaer) )
            sun = sun_spec[f] * exp( - beta[f]*exp(-zt/Za) * Chapman((Rt+zt)/Za,cos_se) - Baer*exp(-zt/Zaer) * Chapman((Rt+zt)/Zaer,cos_se))
            sky_spec[f] += dt * sun  * exp(-taut[f]) * ( 3/16/pi*(1+cosD**2)*beta[f]/Za*exp(-zt/Za) + mie_aerosol_phase_function(cosD)/4/pi*Baer/Zaer*exp(-zt/Zaer) )
    
    alpha = sin(theta_s)*cos(phi-phi_s)* t_surf*sin(theta)/(Rt+t_surf*cos_v+z0)
    theta_se = theta_s-alpha
    if cos(theta_se) > 0: # Reflexion du soleil sur l'eau
        theta_ve = pi - (theta+alpha)
        cos_reflexion = cos(theta_ve)*cos(theta_se) + sin(theta_ve)*sin(theta_se)*cos(phi-phi_s)
        coef = sun_reflexion_phase_function(cos_reflexion)/4/pi
        C = cos(theta_se)
        SQRT = sqrt(1-(sin(theta_se)/1.34)**2)
        fresnel_R = (  ( (C-1.34*SQRT) / (C+1.34*SQRT) )**2 + ( (SQRT-1.34*C) / (SQRT+1.34*C) )**2 )/2
        for f in range(3):
            sun = sun_spec[f] * exp( - beta[f] * Chapman(Rt/Za,cos_se) - Baer* Chapman(Rt/Zaer,cos_se))
            sky_spec[f] += coef * fresnel_R * sun * max_surface_sca[f] * exp(-taut[f])
          
    dt = (t_max-t_surf)/n
    for it in range(n):
        t = t_surf + (it+0.5)*dt
        zt = -Rt + sqrt(Rt**2 + z0**2 + 2*Rt*z0 + 2*(Rt+z0)*t*cos_v + t**2 )
        alpha = sin(theta_s)*cos(phi-phi_s)* t*sin(theta)/(Rt+t*cos_v+z0)
        cos_se = cos(theta_s-alpha)
        for f in range(3):
            taut[f] += dt*( beta[f]/Za*exp(-zt/Za) + Baer/Zaer*exp(-zt/Zaer) )
            sun = sun_spec[f] * exp( - beta[f]*exp(-zt/Za) * Chapman((Rt+zt)/Za,cos_se))
            sky_spec[f] += dt * sun * max_surface_sca[f] * exp(-taut[f]) * ( 3/16/pi*(1+cosD**2)*beta[f]/Za*exp(-zt/Za) )
    
       
@njit
def poly(t,a,b,c,d):
    return (2*b + t*(c-a) + t**2*(2*a-5*b+4*c-d) + t**3*(3*(b-c)+d-a))/2
@njit
def bicubic_interpolation(f_,k,l,a,b):
    b_1= poly(a,f_[k-1,l-1],f_[k  ,l-1],f_[k+1,l-1],f_[k+2,l-1])
    b0 = poly(a,f_[k-1,l  ],f_[k  ,l  ],f_[k+1,l  ],f_[k+2,l  ])
    b1 = poly(a,f_[k-1,l+1],f_[k  ,l+1],f_[k+1,l+1],f_[k+2,l+1])
    b2 = poly(a,f_[k-1,l+2],f_[k  ,l+2],f_[k+1,l+2],f_[k+2,l+2])
    return poly(b,b_1,b0,b1,b2)

@njit(parallel=True)
def view_precise(res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,surface0,surface,sca,image,z0,x0,y0,theta0,phi0,FOV,mag_range):
    nz,nx,ny = k_ext.shape
    nv,nh,_ = image.shape
    z_real = res*z0
    horizon_theta = acos(-sqrt(2*Rt*z_real+z_real**2)/Rt)
    min_ray = 10**(-mag_range)
    for j in prange(nv):
        for i in range(nh):
            z,x,y = z0,x0,y0
            theta,phi = projection_FOV(theta0,phi0,nh,nv,FOV,i,j)
            dz,dx,dy = cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)
            
            sky_spec = np.zeros(3,dtype=floatype)
            ray = np.ones(3,dtype=floatype)
                
            if dx>0 and x<0.:
                z-=dz*x/dx ; y-=dy*x/dx ; x=0.
            elif dx<0 and x>nx-2:
                z+=dz*(nx-2-x)/dx ; y+=dy*(nx-2-x)/dx ; x=nx-2
            if dy>0 and y<0.:
                z-=dz*y/dy ; x-=dx*y/dy ; y=0.
            elif dy<0 and y>ny-2:
                z+=dz*(ny-2-y)/dy ; x+=dx*(ny-2-y)/dy ; y=ny-2
            if dz>0 and z<0.:
                x-=dx*z/dz ; y-=dy*z/dz ; z=0.
            elif dz<0 and z>nz-2:
                x+=dx*(nz-2-z)/dz ; y+=dy*(nz-2-z)/dz ; z=nz-2
            
            while z>=0. and z<nz-1 and x>=0. and x<nx-1 and y>=0. and y<ny-1:
                iz=int(z) ; ix=int(x) ; iy=int(y)
                if dz<0 and z==iz : iz-=1
                if dx<0 and x==ix : ix-=1
                if dy<0 and y==iy : iy-=1
                if iz<0 or ix<0 or iy<0: break
                D = 2
                if dz>0:
                    D = min(D,(1-z+iz)/dz)
                elif dz<0:
                    D = min(D,(-z+iz)/dz)
                if dx>0:
                    D = min(D,(1-x+ix)/dx)
                elif dx<0:
                    D = min(D,(-x+ix)/dx)
                if dy>0:
                    D = min(D,(1-y+iy)/dy)
                elif dy<0:
                    D = min(D,(-y+iy)/dy)
                
                if k_ext[iz,ix,iy]>0. or k_ext[iz+1,ix,iy]>0. or k_ext[iz,ix+1,iy]>0. or k_ext[iz,ix,iy+1]>0. or k_ext[iz+1,ix+1,iy]>0. or k_ext[iz+1,ix,iy+1]>0. or k_ext[iz,ix+1,iy+1]>0. or k_ext[iz+1,ix+1,iy+1]>0.:
                    mean_k_ext = mean_ray(z-iz,x-ix,y-iy,dz*D,dx*D,dy*D,k_ext[iz,ix,iy],k_ext[iz+1,ix,iy],k_ext[iz,ix+1,iy],k_ext[iz,ix,iy+1],k_ext[iz,ix+1,iy+1],k_ext[iz+1,ix,iy+1],k_ext[iz+1,ix+1,iy],k_ext[iz+1,ix+1,iy+1])
                    frac = exp( -D*mean_k_ext )
                    for f in range(3):
                        mean_sca = mean_ray(z-iz,x-ix,y-iy,dz*D,dx*D,dy*D,sca[iz,ix,iy,f],sca[iz+1,ix,iy,f],sca[iz,ix+1,iy,f],sca[iz,ix,iy+1,f],sca[iz,ix+1,iy+1,f],sca[iz+1,ix,iy+1,f],sca[iz+1,ix+1,iy,f],sca[iz+1,ix+1,iy+1,f])
                        image[j,i,f] += ray[f]*mean_sca*(1-frac)/mean_k_ext
                        ray[f] *= frac
                    if ray[0]+ray[1]+ray[2]<min_ray:
                        break
                z+=dz*D ; x+=dx*D ; y+=dy*D
                
            if ray[0]+ray[1]+ray[2]>min_ray:
                if theta < horizon_theta:
                    sky_spec_round_earth_aerosols(z_real,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,theta,phi,sky_spec)
                else:
                    x = x0+dx*z0/abs(dz) ; y = y0+dy*z0/abs(dz)
                    if x>=0. and x<nx-1 and y>=0. and y<ny-1:
                        inside = True
                    else:
                        x= nx-1 - abs( (x)%(2*nx-2) -nx+1 ) ; y= ny-1 -abs( (y)%(2*ny-2) -ny+1 )
                        inside = False
                        
                    ix=int(x) ; iy=int(y)
                    b = x-ix ; c = y-iy
                    surface_sca = np.zeros(3,dtype=floatype)
                    if surface0[ix,iy,0] < 0.:
                        for f in range(3):
                            if inside:
                                surface_sca[f] = -min(0.,bicubic_interpolation(surface[:,:,f],ix,iy,b,c))
                            else:
                                surface_sca[f] = -min(0.,bicubic_interpolation(surface0[:,:,f],ix,iy,b,c))
                        sky_spec_round_earth_aerosols_ocean(z_real,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,theta,phi,sky_spec,surface_sca)
                    else:
                        for f in range(3):
                            if inside:
                                surface_sca[f]  = max(0.,bicubic_interpolation(surface[:,:,f],ix,iy,b,c))
                            else:
                                surface_sca[f]  = max(0.,bicubic_interpolation(surface0[:,:,f],ix,iy,b,c))
                        sky_spec_round_earth_aerosols_surface(z_real,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,theta,phi,sky_spec,surface_sca)
                for f in range(3):
                    image[j,i,f] += ray[f] * sky_spec[f]
                    
            '''
            https://en.wikipedia.org/w/index.php?title=HSL_and_HSV'
            '''
            R,G,B = image[j,i,0],image[j,i,1],image[j,i,2]
            M = max(R,G,B)
            m = min(R,G,B)
            C = M-m
            if C==0.:
                if R>0.:
                    for f in range(3):
                        image[j,i,f] = 1+log10(image[j,i,f])/mag_range
            else:
                if M==R:
                    H = ((G-B)/C)%6
                elif M==G:
                    H = (B-R)/C +2
                elif M==B:
                    H = (R-G)/C +4
                Luma0 = 0.2989*R + 0.5871*G + 0.1140*B
                Luma = 1+log10(Luma0)/mag_range
                if Luma>1.: Luma=1.
                C*=Luma/Luma0
                X = C*(1-abs(H%2-1))
                if H<1:
                    R1,G1,B1 = C,X,0.
                elif H<2:
                    R1,G1,B1 = X,C,0.
                elif H<3:
                    R1,G1,B1 = 0.,C,X
                elif H<4:
                    R1,G1,B1 = 0.,X,C
                elif H<5:
                    R1,G1,B1 = X,0.,C
                elif H<6:
                    R1,G1,B1 = C,0.,X
                m = Luma - (0.2989*R1 + 0.5871*G1 + 0.1140*B1)
                image[j,i,0],image[j,i,1],image[j,i,2] = R1+m,G1+m,B1+m
            for f in range(3):
                if image[j,i,f]<0.:
                    image[j,i,f]=0.
                elif image[j,i,f]>1.:
                    image[j,i,f]=1.
                
            
@njit(parallel=True)
def linear_expand(im,imX,coef):
    nv,nh,_ = im.shape
    height,width,_ = imX.shape
    for j in prange(height):
        for i in range(width):
            dv = (height-1-j) %coef ; dh = i %coef
            iv = (height-1-j)//coef ; ih = i//coef
            if dv==0 and dh==0:
                for f in range(3):
                    imX[j,i,f] = int(im[iv,ih,f]*255)
            elif dv==0:
                for f in range(3):
                    imX[j,i,f] = int(((coef-dh)*im[iv,ih,f] + dh*im[iv,ih+1,f])/coef*255)
            elif dh==0:
                for f in range(3):
                    imX[j,i,f] = int(((coef-dv)*im[iv,ih,f] + dv*im[iv+1,ih,f])/coef*255)
            else:
                for f in range(3):
                    imX[j,i,f] = int(((coef-dv)*(coef-dh)*im[iv,ih,f] + dv*(coef-dh)*im[iv+1,ih,f]  + dh*(coef-dv)*im[iv,ih+1,f] + dv*dh*im[iv+1,ih+1,f])/coef**2*255)



#%%
theta_s,phi_s = les_theta_s[0] , les_phi_s[0]
z0 = 1.1
x0 = nx/2
y0 = ny/2
theta0 = 100 *pi/180
phi0 = 90 *pi/180
FOV = 90 *pi/180
nh = 200#512
nv = 100#256
screen_expansion = 8
# nh = 600#512
# nv = 300#256
# screen_expansion = 3
width,height = screen_expansion*(nh-1)+1 , screen_expansion*(nv-1)+1
im=np.zeros((nv,nh,3),dtype=floatype)
imX=np.zeros((height,width,3),dtype=np.uint8)
imF=np.zeros((height,width,3),dtype=floatype)


speed = 3
rotate_speed = 2*pi/180
intensity = 0.05
opacity = 6.
tauG,tauRB = 0.5,3.
it = 0

SCREEN = pygame.display.set_mode(( width,height), vsync=True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

pygame.display.init()

show_sparse = [True,False,False,False]
for v,var in enumerate(sparse_names):
    if var=='rcloud':
        if show_sparse[v]:
            fill_kext_from_sparse(list_coords_sparse[v],list_kext_sparse[v],k_ext,list_time_indexes[v][it],list_time_indexes[v][it+1])
theta_s,phi_s = les_theta_s[it] , les_phi_s[it]
# solar_income_precise(res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,sca)
solar_income_vertical_shadow(sun_spec,k_ext,sca,surface0,surface)
view_precise(res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,surface0,surface,sca,im,z0,x0,y0,theta0,phi0,FOV,mag_range)
linear_expand(im,imX,screen_expansion)
surf = pygame.image.frombuffer(imX.tobytes(), (width,height), "RGB")
SCREEN.blit(surf, (0, 0))
pygame.display.flip()

#%%
def fullres():
    imF[:] = 0.
    view_precise(res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,surface0,surface,sca,imF,z0,x0,y0,theta0,phi0,FOV,mag_range)
    imX = np.array(imF[::-1]*255,dtype=np.uint8)
    surf = pygame.image.frombuffer(imX.tobytes(), (width,height), "RGB")
    SCREEN.blit(surf, (0, 0))
    pygame.display.flip()
    
def pause():
    loop = 1
    while loop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                loop = 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    loop = 0
        clock.tick(200)
    
pygame.init()
pygame.mouse.set_cursor((8,8),(0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0))
clock = pygame.time.Clock()
FRAME = 0
run = True
avg_fps=0
update_data=False
update_view=False
while run:
    clock.tick(40)
    fps = clock.get_fps()
    pygame.display.set_caption(simu+' - {:02d}h{:02d}'.format(int(time[it]),int(time[it]%1*60))+" : FPS %s " % round(fps, 2) + " - Mean %s " % round(avg_fps, 2))
    # pygame.event.pump()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_z]: x0 += speed*cos(phi0) ; y0 += speed*sin(phi0) ; update_view=True
    if keys[pygame.K_s]: x0 -= speed*cos(phi0) ; y0 -= speed*sin(phi0) ; update_view=True
    if keys[pygame.K_q]: x0 -= speed*sin(phi0) ; y0 += speed*cos(phi0) ; update_view=True
    if keys[pygame.K_d]: x0 += speed*sin(phi0) ; y0 -= speed*cos(phi0) ; update_view=True
    if keys[pygame.K_SPACE]: z0 += speed ; update_view=True
    if keys[pygame.K_LSHIFT] and z0>speed+1: z0 -= speed ; update_view=True
    if keys[pygame.K_DOWN] and theta0<pi:
        theta0 += rotate_speed ; update_view=True
        if theta0>pi: theta0=pi
    if keys[pygame.K_UP] and theta0>0:
        theta0 -= rotate_speed ; update_view=True
        if theta0<0: theta0=0.
    if keys[pygame.K_RIGHT]: phi0 -= rotate_speed ; update_view=True
    if keys[pygame.K_LEFT]: phi0 += rotate_speed ; update_view=True
    if keys[pygame.K_e] and it<nt-1: it += 1 ; update_data=True
    if keys[pygame.K_a] and it >0  : it -= 1 ; update_data=True
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT: run = False ; pygame.quit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_e and it<nt-1: it+=1 ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_a and it>0   : it-=1 ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_f: fullres()#; pause()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: run = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p: pause()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_i: intensity*=1.2 ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_k: intensity/=1.2 ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_o: opacity*=1.2 ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_l: opacity/=1.2 ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_g: tauG*=1.2 ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_h: tauG/=1.2 ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_b: tauRB*=1.2 ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_n: tauRB/=1.2 ; update_data=True
        if event.type == pygame.MOUSEMOTION and event.rel != (0,0):
            phi0 -= event.rel[0] /300
            theta0 += event.rel[1] /300
            if theta0>pi: theta0=pi
            if theta0<0: theta0=0.
            pygame.mouse.set_pos(SCREEN.get_rect().center)
            update_view=True
            
        if event.type == pygame.KEYDOWN and event.key == pygame.K_1: show_sparse[0]=not(show_sparse[0]) ; print(sparse_names[0],str(show_sparse[0])) ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_2: show_sparse[1]=not(show_sparse[1]) ; print(sparse_names[1],str(show_sparse[1])) ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_3: show_sparse[2]=not(show_sparse[2]) ; print(sparse_names[2],str(show_sparse[2])) ; update_data=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_4: show_sparse[3]=not(show_sparse[3]) ; print(sparse_names[3],str(show_sparse[3])) ; update_data=True
        
    if update_data:
        sca[:] = 0.
        k_ext[:] = 0.
        for v,var in enumerate(sparse_names):
            if show_sparse[v]:
                if var=='rcloud' or var=='rprecip':
                    fill_kext_from_sparse(list_coords_sparse[v],list_kext_sparse[v],k_ext,list_time_indexes[v][it],list_time_indexes[v][it+1])
                
        theta_s,phi_s = les_theta_s[it] , les_phi_s[it]
        # solar_income_precise(res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,sca)
        solar_income_vertical_shadow(sun_spec,k_ext,sca,surface0,surface)
        
        # for v,var in enumerate(sparse_names):
        #     if show_sparse[v]:
        #         if var!='rcloud' and var!='rprecip':
        #             fill_sca_and_kext_from_sparse_variable(list_coords_sparse[v],list_kext_sparse[v],sca,k_ext,list_time_indexes[v][it],list_time_indexes[v][it+1],intensity,opacity,tauG,tauRB)
        # surface_shadow(k_ext,theta_s,phi_s,surface0,surface)
        update_data=False
        update_view=True

    if update_view:
        im[:] = 0.
        view_precise(res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,surface0,surface,sca,im,z0,x0,y0,theta0,phi0,FOV,mag_range)
        linear_expand(im,imX,screen_expansion)
        
        surf = pygame.image.frombuffer(imX.tobytes(), (width,height), "RGB")
        SCREEN.blit(surf, (0, 0))
        avg_fps = (avg_fps*FRAME + fps)/(FRAME+1)
        FRAME += 1
        pygame.display.flip()
        update_view=False

#%%
from cupyx.profiler import benchmark
# sparse_sca_t = sparse.load_npz('r_cloud_sparse_AMMA.npz')
# # def get_sca(sparse_sca_t,it):
# sca_CPU = sparse_sca_t[-1,:,:,:].todense()*10**-6 
# sca_GPU =  cp.asarray( sca_CPU,dtype=floatype)

# def reset(x_GPU):
#     # x_GPU = cp.zeros_like(x_GPU)
#     # x_GPU *= 0
#     x_GPU[:] = 0
# print(benchmark(reset,(sca_GPU,), n_repeat=100))
# # print(benchmark(get_sca,(sparse_sca_t,0), n_repeat=10))

# v=0
# it = 0
# def fill(sca,k_ext):
#     sca[:] = 0.
#     k_ext[:] = 0.
#     print(list_time_indexes[v][it+1]-list_time_indexes[v][it])
#     fill_sca_and_kext_from_sparse_variable(list_coords_sparse[v],list_kext_sparse[v],sca,k_ext,list_time_indexes[v][it],list_time_indexes[v][it+1],intensity,opacity,tauG,tauRB)
# print(benchmark(fill,(sca,k_ext,), n_repeat=5))

def income():
    # solar_income_vertical_shadow(sun_spec,k_ext[0],sca[0],surface0,surface)
    # solar_income_vertical_shadow(sun_spec,k_ext,sca,surface0,surface)
    # solar_income_precise(res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,sca)
    # surface_shadow(k_ext,theta_s,phi_s,surface0,surface)
    solar_income_vertical_shadow_mipmaps(k_ext,sca,surface0,surface)
print(benchmark(income,(), n_repeat=5))