# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:35:40 2022

@author: Nathan
"""


import numpy as np
import cupy as cp
from numba import cuda,njit#,prange
from math import cos,sin,tan,sqrt,exp,log10,pi,acos,atan2,erfc,ceil#,asin,erf,log,
import sparse
import matplotlib.image as mpimg

floatype = np.float32#np.float64#
intype = np.int16
complextype = np.complex64#np.complex128#

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
mempool.free_all_blocks()
pinned_mempool.free_all_blocks()
rad = pi/180

def day_of_year(day,month):
    return np.sum(np.array([31,28,31,30,31,30,31,31,30,31,30,31])[:month-1]) + day

#%%
# simu='BOMEX' ; surface_type = 'ocean' ; res = 100 ; time = np.linspace(0+1,120,120) ; day = day_of_year(10,6) ; lon=-53 ; lat=13. ; timezone=-4
# simu='AMMA' ; surface_type = 'image' ; res = 200  ; time = np.linspace(12+1/60,18,360) ; day = day_of_year(10,6) ; lon=2 ; lat=13.7 ; timezone=1
simu='AMMA_end' ; surface_type = 'image' ; res = 200  ; time = np.linspace(16+40/60,18,81) ; day = day_of_year(10,6) ; lon=2 ; lat=13.7 ; timezone=1
# simu='AMMHF' ; surface_type = 'image' ; res = 200  ; time = np.linspace(12+1/12,18,72) ; day = day_of_year(10,6) ; lon=2 ; lat=13.7 ; timezone=1
# simu='AMA0W' ; surface_type = 'image' ; res = 200  ; time = np.linspace(12+1/60,18,360) ; day = day_of_year(10,6) ; lon=2 ; lat=13.7 ; timezone=1
# simu='LBA' ; surface_type = 'image' ; res = 200 ; time = np.linspace(9.5+1/60,19.5,600) ; day = day_of_year(23,2) ; lon=-60.1 ; lat=-3.15 ; timezone=-4
# simu='LBALF' ; surface_type = 'image' ; res = 200 ; time = np.linspace(9.5+1/1,15.5,6) ; day = day_of_year(23,2) ; lon=-60.1 ; lat=-3.15 ; timezone=-4
# simu='EXP01' ; surface_type = 'ocean' ; res = 100  ; time = np.linspace(13,18,6) ; day = day_of_year(2,2) ; lon=-59 ; lat=12. ; timezone=-4

# simu='cheminee' ; surface_type = 'black' ; res = 0.1 ; time = np.linspace(13,13+83.27/3600,500) ; day = day_of_year(16,1) ; lon=7 ; lat=46 ; timezone=1
# simu='myclouds' ; surface_type = 'black' ; res = 30 ; time = np.linspace(12+1/60,18,100) ; day = day_of_year(10,6) ; lon=2 ; lat=13.7 ; timezone=1

# simu='CloudBotany1' ; surface_type = 'ocean' ; res = 100  ; time = np.linspace(13,15,3) ; day = day_of_year(2,2) ; lon=-59 ; lat=12. ; timezone=-4
 

sparse_names = ['rcloud']
sparse_names += ['rprecip']
sparse_names += ['thva']
# sparse_names += ['w']
list_sparse = [sparse.load_npz(var+'_sparse_'+simu+'.npz')[:,:] for var in sparse_names ]

if surface_type == 'image':
    surface0 = cp.asarray(np.flipud(mpimg.imread("surface_"+simu+".png")[:,:,:3]),dtype=floatype) 
elif surface_type == 'ocean':
    surface0 = -cp.ones(list_sparse[0].shape[2:]+(3,),dtype=floatype)
elif surface_type == 'black':
    surface0 = 0.0001*cp.ones(list_sparse[0].shape[2:]+(3,),dtype=floatype)
    # surface0[0,:,:] = 1 ;surface0[:,0,:] = 1 ; surface0[-1,:,:] = 1 ;surface0[:,-1,:] = 1 # bordure de domaine en blanc

#%%
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

#%%
nvar = len(sparse_names)
nt,nz,nx,ny = list_sparse[0].shape
list_ns = [sparse_.data.shape[0] for sparse_ in list_sparse ]

lamda = cp.array( [610.,535.,465.] ,dtype=floatype)
beta = delta*rho0*Za/Ma / (lamda*10**-9)**4 
sun_spec = cp.array( [1.,1.,1.] ,dtype=floatype)

tpb1 = 128
list_bpg1 = [ ceil(ns/tpb1) for ns in list_ns ]

tpb2S = (8,8)
nmax = max(nx,ny,nz)
# bpg2S = (ceil(nx/tpb2S[0]),ceil(ny/tpb2S[1]))
bpg2S = (ceil(nmax/tpb2S[0]),ceil(nmax/tpb2S[1]))

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
@cuda.jit()
def surface_spectral_albedo(surface,mag_range):
    nx,ny,_ = surface.shape
    i,j = cuda.grid(2)
    if i>=0 and i<=nx-1 and j>=0 and j<=ny-1:
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
        surface_spectral_albedo[bpg2S,tpb2S](surface0,mag_range-1)
    else:
        surface_spectral_albedo[bpg2S,tpb2S](surface0,mag_range)
    # print('mean albedo =',2*pi*cp.mean(surface0))
    # mean_surface_sca = cp.mean(surface,axis=(0,1))*1.*sun_spec#/2/pi
        

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
list_coords_sparse = [ cp.asarray(sparse_.coords,dtype=intype) for sparse_ in list_sparse ]
# list_kext_sparse = [ cp.zeros((ns),dtype=floatype) for ns in list_ns ]
list_kext_sparse = []
for v,var in enumerate(sparse_names):
    if var=='rcloud' or var=='rprecip':
        les_z = res*np.array(list_sparse[v].coords[1,:] ,dtype=floatype)
        rho_z = rho0*np.exp(-les_z/Za)
        if var=='rcloud': r_eff = cloud_r_eff
        if var=='rprecip': r_eff = precip_r_eff
        list_kext_sparse.append( cp.asarray( sca_forward_ratio * 3/2 * res/r_eff * rho_z/rho_water * list_sparse[v].data ,dtype=floatype) )

    if var=='w' or var=='thva':
        list_kext_sparse.append( cp.asarray( list_sparse[v].data ,dtype=floatype)) 
        
#%% solar income
@cuda.jit
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
    
    i,j = cuda.grid(2)
    if i>=0 and i<=imax and j>=0 and j<=jmax:
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
        ray = cuda.local.array(3,dtype=floatype)
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

@cuda.jit
def solar_income_vertical(sun_spec,k_ext,surface,sca0):
    nz,nx,ny = k_ext.shape
    i,j = cuda.grid(2)
    if i>=0 and i<=nx-1 and j>=0 and j<=ny-1:
        ray = cuda.local.array(3,dtype=floatype)
        for f in range(3):
            ray[f] = sun_spec[f]
        for iz in range(nz-1,0,-1):
            if k_ext[iz,i,j]>0.:
                frac = exp(-k_ext[iz,i,j])
                for f in range(3):
                    sca0[iz,i,j,f] = (ray[f]+0.01)*(1-frac)#/4/pi
                    ray[f] *= frac
        # for f in range(3):
        #     sca0[0,i,j,f] = surface[i,j,f]*(ray[f]+0.01)#/2/pi

@cuda.jit()
def surface_shadow(k_ext,theta_s,phi_s,surface0,surface):
    nz,nx,ny = k_ext.shape
    D = 1.#sqrt(3)
    dz,dx,dy = D*cos(theta_s), D*sin(theta_s)*cos(phi_s), D*sin(theta_s)*sin(phi_s)
    i,j = cuda.grid(2)
    if i>=0 and i<=nx-1 and j>=0 and j<=ny-1:
        shadow = 1.
        z=0. ; x=i+0.5 ; y=j+0.5
        while z>=0. and z<nz and x>=0. and x<nx and y>=0. and y<ny:
            iz,ix,iy = int(z),int(x),int(y)
            if k_ext[iz,ix,iy]>0:
                shadow *= exp( -D*k_ext[iz,ix,iy] )
            z+=dz ; x+=dx ; y+=dy
        
        for f in range(3):
            surface[i,j,f] = surface0[i,j,f]*shadow
            
@cuda.jit()
def fill_kext_from_sparse(coords_sparse,kext_sparse,k_ext,t1,t2):
    t = cuda.grid(1)
    if t>=t1 and t<t2:
        k_ext[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t]] += kext_sparse[t]

 
@cuda.jit()
def fill_sca_and_kext_from_sparse_variable(coords_sparse,kext_sparse,sca,k_ext,t1,t2,intensity,opacity,tauG,tauRB):
    t = cuda.grid(1)
    if t>=t1 and t<t2:
        if kext_sparse[t] >0:
            sca[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t],1] += kext_sparse[t]*intensity*exp(-kext_sparse[t]/tauG)
            sca[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t],0] += kext_sparse[t]*intensity*exp(-kext_sparse[t]/tauRB)
            k_ext[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t]] += kext_sparse[t]*intensity*opacity
        else:
            sca[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t],1] -= kext_sparse[t]*intensity*exp(kext_sparse[t]/tauG)
            sca[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t],2] -= kext_sparse[t]*intensity*exp(kext_sparse[t]/tauRB)
            k_ext[coords_sparse[1,t],coords_sparse[2,t],coords_sparse[3,t]] -= kext_sparse[t]*intensity*opacity
        
@cuda.jit(device=True)
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

@cuda.jit(device=True)
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

@cuda.jit(device=True)
def Chapman(alpha,cos_s):
    gamma = cos_s*sqrt(alpha/2)
    if gamma < -4:
        return 1e10
    elif gamma < 8:
        return sqrt(pi*alpha/2)* exp(gamma**2) * erfc(gamma)
    else:
        return 1/cos_s
    
@cuda.jit(device=True)
def mie_aerosol_phase_function(cosD):
    D = acos(cosD)
    power=1
    return (3*(1+cosD**2)+ exp((D-pi*3/4)*power)+exp(-(D-pi*2/3)*power*2.5)-2)/15.521057295924523
@cuda.jit(device=True)
def sun_reflexion_phase_function(cosD):
    D = acos(cosD)
    power=3
    return (3*(1+cosD**2)+ exp((D-pi*3/4)*power)+exp(-(D-pi*2/3)*power*2.5)-2)/57101.15796667264

@cuda.jit(device=True)
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
            taut = cuda.local.array(3,dtype=floatype)
        else:
            t += dtj
        for f in range(3):
            taut[f] += dtj*( beta[f]/Za*exp(-zt/Za) + Baer/Zaer*exp(-zt/Zaer) )
            sun = sun_spec[f] * exp( - beta[f]*exp(-zt/Za) * Chapman((Rt+zt)/Za,cos_se) - Baer*exp(-zt/Zaer) * Chapman((Rt+zt)/Zaer,cos_se))
            sky_spec[f] += dtj * sun  * exp(-taut[f]) * ( 3/16/pi*(1+cosD**2)*beta[f]/Za*exp(-zt/Za) + mie_aerosol_phase_function(cosD)/4/pi*Baer/Zaer*exp(-zt/Zaer) )
    if cosD>0.9999:
        for f in range(3):
            sky_spec[f] += sun_spec[f] * exp(-taut[f])
            
@cuda.jit(device=True)
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
    taut = cuda.local.array(3,dtype=floatype)
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
            
@cuda.jit(device=True)
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
    taut = cuda.local.array(3,dtype=floatype)
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
    
       
@cuda.jit(device=True)
def poly(t,a,b,c,d):
    return (2*b + t*(c-a) + t**2*(2*a-5*b+4*c-d) + t**3*(3*(b-c)+d-a))/2
@cuda.jit(device=True)
def bicubic_interpolation(f_,k,l,a,b):
    b_1= poly(a,f_[k-1,l-1],f_[k  ,l-1],f_[k+1,l-1],f_[k+2,l-1])
    b0 = poly(a,f_[k-1,l  ],f_[k  ,l  ],f_[k+1,l  ],f_[k+2,l  ])
    b1 = poly(a,f_[k-1,l+1],f_[k  ,l+1],f_[k+1,l+1],f_[k+2,l+1])
    b2 = poly(a,f_[k-1,l+2],f_[k  ,l+2],f_[k+1,l+2],f_[k+2,l+2])
    return poly(b,b_1,b0,b1,b2)

@cuda.jit
def view_precise2(res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,surface0,surface,sca,image,z0,x0,y0,theta0,phi0,FOV,mag_range):
    nz,nx,ny = k_ext.shape
    nv,nh,_ = image.shape
    z_real = res*z0
    horizon_theta = acos(-sqrt(2*Rt*z_real+z_real**2)/Rt)
    min_ray = 10**(-mag_range)
    # Dfast = 0.5
    j,i = cuda.grid(2)
    if i>=0 and i<=nh-1 and j>=0 and j<=nv-1:
        z,x,y = z0,x0,y0
        theta,phi = projection_FOV(theta0,phi0,nh,nv,FOV,i,j)
        dz,dx,dy = cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)
        if abs(dz)<1e-5: dz=1e-5
        if abs(dx)<1e-5: dx=1e-5
        if abs(dy)<1e-5: dy=1e-5
        
        sky_spec = cuda.local.array(3,dtype=floatype)
        ray = cuda.local.array(3,dtype=floatype)
        for f in range(3):
            ray[f] = 1.
            sky_spec[f] = 0.
            
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
        
        if z>=0 and z<=nz-1 and x>=0 and x<=nx-1 and y>=0 and y<=ny-1:
            sz = 1 if dz>=0 else -1
            sx = 1 if dx>=0 else -1
            sy = 1 if dy>=0 else -1
            iz = int(z) ; ix=int(x) ; iy=int(y)
            if sz==-1 and z>iz: iz+=1
            if sx==-1 and x>ix: ix+=1
            if sy==-1 and y>iy: iy+=1
            
            D = 2.
            Dz = (sz-z+iz)/dz
            if Dz<D:
                D=Dz ; plane=0
            Dx = (sx-x+ix)/dx
            if Dx<D:
                D=Dx ; plane=1
            Dy = (sy-y+iy)/dy
            if Dy<D:
                D=Dy ; plane=2
            z+=dz*D ; x+=dx*D ; y+=dy*D
                                    
            old_zero = True
            old_k_ext = 0.
            old_sca = cuda.local.array(3,dtype=floatype)
            for f in range(3):
                old_sca[f] = 0.
            new_sca = cuda.local.array(3,dtype=floatype)
            # fast = True
            inside = True
            while inside:
                # if fast:
                #     z+=dz*Dfast ; x+=dx*Dfast ; y+=dy*Dfast
                #     if not(z>=0 and z<=nz-1 and x>=0 and x<=nx-1 and y>=0 and y<=ny-1):
                #         inside = False
                #     elif k_ext[round(z),round(x),round(y)]>0:
                #         z-=dz*Dfast ; x-=dx*Dfast ; y-=dy*Dfast
                #         fast = False
                # else:
                    if plane==0:
                        D = sz/dz
                        Dx = (sx-x+ix)/dx
                        if Dx<D:
                            D=Dx ; plane=1
                        Dy = (sy-y+iy)/dy
                        if Dy<D:
                            D=Dy ; plane=2
                    
                    elif plane==1:
                        D = sx/dx
                        Dz = (sz-z+iz)/dz
                        if Dz<D:
                            D=Dz ; plane=0
                        Dy = (sy-y+iy)/dy
                        if Dy<D:
                            D=Dy ; plane=2
                            
                    elif plane==2:
                        D = sy/dy
                        Dz = (sz-z+iz)/dz
                        if Dz<D:
                            D=Dz ; plane=0
                        Dx = (sx-x+ix)/dx
                        if Dx<D:
                            D=Dx ; plane=1
                    
                    z+=dz*D ; x+=dx*D ; y+=dy*D
                    
                    if plane==0:
                        iz+=sz
                        if iz==0 or iz==nz-1: inside=False
                        if k_ext[iz,ix,iy]>0 or k_ext[iz,ix+sx,iy]>0 or k_ext[iz,ix,iy+sy]>0 or k_ext[iz,ix+sx,iy+sy]>0:
                            new_zero = False
                            b = abs(x-ix) ; c = abs(y-iy) ; a=0.
                            b_ = 1-b ; c_ = 1-c
                            new_k_ext = b_*(c*k_ext[iz,ix,iy+sy] + c_*k_ext[iz,ix,iy]) + b*(c*k_ext[iz,ix+sx,iy+sy] + c_*k_ext[iz,ix+sx,iy])
                            for f in range(3):
                                new_sca[f]  = b_*(c*sca[iz,ix,iy+sy,f] + c_*sca[iz,ix,iy,f]) + b*(c*sca[iz,ix+sx,iy+sy,f] + c_*sca[iz,ix+sx,iy,f])
                        else:
                            new_zero = True
                    if plane==1:
                        ix+=sx
                        if ix==0 or ix==nx-1: inside=False
                        if k_ext[iz+sz,ix,iy]>0 or k_ext[iz+sz,ix,iy+sy]>0 or k_ext[iz,ix,iy]>0 or k_ext[iz,ix,iy+sy]>0:
                            new_zero = False
                            a = abs(z-iz) ; c = abs(y-iy) ; b=0.
                            a_ = 1-a ; c_ = 1-c
                            new_k_ext = a_*(c*k_ext[iz,ix,iy+sy] + c_*k_ext[iz,ix,iy]) + a*(c*k_ext[iz+sz,ix,iy+sy] + c_*k_ext[iz+sz,ix,iy])
                            for f in range(3):
                                new_sca[f]  = a_*(c*sca[iz,ix,iy+sy,f] + c_*sca[iz,ix,iy,f]) + a*(c*sca[iz+sz,ix,iy+sy,f] + c_*sca[iz+sz,ix,iy,f])
                        else:
                            new_zero = True
                    if plane==2:
                        iy+=sy
                        if iy==0 or iy==ny-1: inside=False
                        if k_ext[iz+sz,ix,iy]>0 or k_ext[iz+sz,ix+sx,iy]>0 or k_ext[iz,ix,iy]>0 or k_ext[iz,ix+sx,iy]>0:
                            new_zero = False
                            a = abs(z-iz) ; b = abs(x-ix) ; c=0.
                            a_ = 1-a ; b_ = 1-b
                            new_k_ext = a_*(b*k_ext[iz,ix+sx,iy] + b_*k_ext[iz,ix,iy]) + a*(b*k_ext[iz+sz,ix+sx,iy] + b_*k_ext[iz+sz,ix,iy])
                            for f in range(3):
                                new_sca[f]  = a_*(b*sca[iz,ix+sx,iy,f] + b_*sca[iz,ix,iy,f]) + a*(b*sca[iz+sz,ix+sx,iy,f] + b_*sca[iz+sz,ix,iy,f])
                        else:
                            new_zero = True
                            
                    if not(old_zero) or not(new_zero):
                        if new_zero:
                            new_k_ext = 0.
                            for f in range(3):
                                new_sca[f] = 0.
                        
                        if old_zero ^ new_zero:
                            if plane==0:
                                plane=0
                                # mean_k_ext = mean_ray(a,b,c,dz*D,dx*D,dy*D,k_ext[iz,ix,iy],k_ext[iz-sz,ix,iy],k_ext[iz,ix+sx,iy],k_ext[iz,ix,iy+sy],k_ext[iz,ix+sx,iy+sy],k_ext[iz-sz,ix,iy+sy],k_ext[iz-sz,ix+sx,iy],k_ext[iz-sz,ix+sx,iy+sy])
                            # if plane==1:
                            #     mean_k_ext = mean_ray(a,b,c,dz*D,dx*D,dy*D,k_ext[iz,ix,iy],k_ext[iz+sz,ix,iy],k_ext[iz,ix-sx,iy],k_ext[iz,ix,iy+sy],k_ext[iz,ix-sx,iy+sy],k_ext[iz+sz,ix,iy+sy],k_ext[iz+sz,ix-sx,iy],k_ext[iz+sz,ix-sx,iy+sy])
                            # if plane==2:
                            #     mean_k_ext = mean_ray(a,b,c,dz*D,dx*D,dy*D,k_ext[iz,ix,iy],k_ext[iz+sz,ix,iy],k_ext[iz,ix+sx,iy],k_ext[iz,ix,iy-sy],k_ext[iz,ix+sx,iy-sy],k_ext[iz+sz,ix,iy-sy],k_ext[iz+sz,ix+sx,iy],k_ext[iz+sz,ix+sx,iy-sy])
                            # frac = exp( -D*mean_k_ext )
                            # for f in range(3):
                            #     if plane==0:
                            #         mean_sca = mean_ray(a,b,c,dz*D,dx*D,dy*D,sca[iz,ix,iy,f],sca[iz-sz,ix,iy,f],sca[iz,ix+sx,iy,f],sca[iz,ix,iy+sy,f],sca[iz,ix+sx,iy+sy,f],sca[iz-sz,ix,iy+sy,f],sca[iz-sz,ix+sx,iy,f],sca[iz-sz,ix+sx,iy+sy,f])
                            #     if plane==1:
                            #         mean_sca = mean_ray(a,b,c,dz*D,dx*D,dy*D,sca[iz,ix,iy,f],sca[iz+sz,ix,iy,f],sca[iz,ix-sx,iy,f],sca[iz,ix,iy+sy,f],sca[iz,ix-sx,iy+sy,f],sca[iz+sz,ix,iy+sy,f],sca[iz+sz,ix-sx,iy,f],sca[iz+sz,ix-sx,iy+sy,f])
                            #     if plane==2:
                            #         mean_sca = mean_ray(a,b,c,dz*D,dx*D,dy*D,sca[iz,ix,iy,f],sca[iz+sz,ix,iy,f],sca[iz,ix+sx,iy,f],sca[iz,ix,iy-sy,f],sca[iz,ix+sx,iy-sy,f],sca[iz+sz,ix,iy-sy,f],sca[iz+sz,ix+sx,iy,f],sca[iz+sz,ix+sx,iy-sy,f])
                            #     image[j,i,f] += ray[f]*mean_sca*(1-frac)/mean_k_ext
                            #     ray[f] *= frac
                        else: 
                            mean_k_ext = (new_k_ext+old_k_ext)/2
                            frac = exp( -D*mean_k_ext )
                            for f in range(3):
                                # mean_sca = (new_sca[f]+old_sca[f])/2
                                # image[j,i,f] += ray[f]*mean_sca*(1-frac)/mean_k_ext
                                alpha = (new_k_ext*new_sca[f]+old_k_ext*old_sca[f]) / (new_k_ext**2+old_k_ext**2)
                                image[j,i,f] += ray[f]*alpha*(1-frac)
                                ray[f] *= frac
                        
                        for f in range(3):
                            old_sca[f] = new_sca[f]
                        old_k_ext = new_k_ext
                        
                        if ray[0]+ray[1]+ray[2]<min_ray:
                            inside=False
                    
                    old_zero = new_zero
                    # if new_zero:
                    #     fast = True
            
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
                surface_sca = cuda.local.array(3,dtype=floatype)
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

@cuda.jit
def view_precise(res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,surface0,surface,sca,image,z0,x0,y0,theta0,phi0,FOV,mag_range):
    nz,nx,ny = k_ext.shape
    nv,nh,_ = image.shape
    z_real = res*z0
    horizon_theta = acos(-sqrt(2*Rt*z_real+z_real**2)/Rt)
    min_ray = 10**(-mag_range)
    j,i = cuda.grid(2)
    if i>=0 and i<=nh-1 and j>=0 and j<=nv-1:
        z,x,y = z0,x0,y0
        theta,phi = projection_FOV(theta0,phi0,nh,nv,FOV,i,j)
        dz,dx,dy = cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)
        
        sky_spec = cuda.local.array(3,dtype=floatype)
        ray = cuda.local.array(3,dtype=floatype)
        for f in range(3):
            ray[f] = 1.
            sky_spec[f] = 0.
            
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
                surface_sca = cuda.local.array(3,dtype=floatype)
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
                
@cuda.jit()
def smooth(im,imS,d):
    nv,nh,_ = im.shape
    j,i = cuda.grid(2)
    if i>=0 and i<=nh-d and j>=0 and j<=nv-d:
        for f in range(3):
            for dj in range(d):
                for di in range(d):
                    imS[j,i,f] += im[j+dj,i+di,f]
            imS[j,i,f] /= d**2
            
@cuda.jit()
def linear_expand(im,imX,coef):
    nv,nh,_ = im.shape
    height,width,_ = imX.shape
    j,i = cuda.grid(2)
    if i>=0 and i<=width-1 and j>=0 and j<=height-1:
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
import pygame
from pygame import RLEACCEL

theta_s,phi_s = les_theta_s[0] , les_phi_s[0]
z0 = 1.1
x0 = nx/2
y0 = ny/2
theta0 = 100 *pi/180
phi0 = 90 *pi/180
FOV = 90 *pi/180
nh = 400#512
nv = 200#256
screen_expansion = 4
# nh = 600#512
# nv = 300#256
# screen_expansion = 3
width,height = screen_expansion*(nh-1)+1 , screen_expansion*(nv-1)+1
im=cp.zeros((nv,nh,3),dtype=floatype)
imS=cp.zeros((nv,nh,3),dtype=floatype)
imX=cp.zeros((height,width,3),dtype=cp.uint8)
imF=cp.zeros((height,width,3),dtype=floatype)

tpb2 = (4,4)
# tpb2 = (8,16)
bpg2 = (ceil(nv/tpb2[0]),ceil(nh/tpb2[1]))
tpb2X = (4,4)
# tpb2X = (8,16)
bpg2X = (ceil(height/tpb2X[0]),ceil(width/tpb2X[1]))

sca = cp.zeros((nz,nx,ny,3),dtype=floatype)
k_ext = cp.zeros((nz,nx,ny),dtype=floatype)
surface = cp.copy(surface0)
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
            fill_kext_from_sparse[list_bpg1[v],tpb1](list_coords_sparse[v],list_kext_sparse[v],k_ext,list_time_indexes[v][it],list_time_indexes[v][it+1])
theta_s,phi_s = les_theta_s[it] , les_phi_s[it]
solar_income_precise[bpg2S,tpb2S](res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,sca)
view_precise[bpg2,tpb2](res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,surface0,surface,sca,im,z0,x0,y0,theta0,phi0,FOV,mag_range)
linear_expand[bpg2X,tpb2X](im,imX,screen_expansion)
surf = pygame.image.frombuffer(imX.tobytes(), (width,height), "RGB")
SCREEN.blit(surf, (0, 0))
pygame.display.flip()

#%%
def fullres():
    imF[:] = 0.
    view_precise[bpg2X,tpb2X](res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,surface0,surface,sca,imF,z0,x0,y0,theta0,phi0,FOV,mag_range)
    imX = cp.asarray(imF[::-1]*255,dtype=cp.uint8)
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
                    fill_kext_from_sparse[list_bpg1[v],tpb1](list_coords_sparse[v],list_kext_sparse[v],k_ext,list_time_indexes[v][it],list_time_indexes[v][it+1])
                
        theta_s,phi_s = les_theta_s[it] , les_phi_s[it]
        solar_income_precise[bpg2S,tpb2S](res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,sca)
        
        for v,var in enumerate(sparse_names):
            if show_sparse[v]:
                if var!='rcloud' and var!='rprecip':
                    fill_sca_and_kext_from_sparse_variable[list_bpg1[v],tpb1](list_coords_sparse[v],list_kext_sparse[v],sca,k_ext,list_time_indexes[v][it],list_time_indexes[v][it+1],intensity,opacity,tauG,tauRB)
        surface_shadow[bpg2S,tpb2S](k_ext,theta_s,phi_s,surface0,surface)
        update_data=False
        update_view=True

    if update_view:
        im[:] = 0.
        # imS[:] = 0.
        view_precise[bpg2,tpb2](res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,surface0,surface,sca,im,z0,x0,y0,theta0,phi0,FOV,mag_range)
        # smooth[bpg2,tpb2](im,imS,3)
        linear_expand[bpg2X,tpb2X](im,imX,screen_expansion)
        
        surf = pygame.image.frombuffer(imX.tobytes(), (width,height), "RGB")
        SCREEN.blit(surf, (0, 0))
        avg_fps = (avg_fps*FRAME + fps)/(FRAME+1)
        FRAME += 1
        pygame.display.flip()
        update_view=False

#%%
# from cupyx.profiler import benchmark
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

# it = 11
# tpb1 = 128
# bpg1 = ceil(ns/tpb1)
# def fill(sca,k_ext):
#     print(time_indexes[it+1]-time_indexes[it])
#     fill_sca_and_kext_from_sparse[bpg1,tpb1](coords_sparse,sca_kext_sparse,sca,k_ext,time_indexes[it],time_indexes[it+1])
# print(benchmark(fill,(sca,k_ext,), n_repeat=5))

# def income():
#     solar_income_precise[bpg2S,tpb2S](res,Rt,Za,beta,Zaer,Baer,sun_spec,theta_s,phi_s,k_ext,sca)
# print(benchmark(income,(), n_repeat=5))