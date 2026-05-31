# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:15:25 2023

@author: TALHA
"""

import numpy as np
from numpy import cos,sin,exp,zeros,max
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import sys
import warnings
warnings.filterwarnings('ignore')

s_exact = lambda x,y,t: cos(x*y)*exp(t)
i_exact = lambda x,y,t: exp(x)*sin(y)*cos(t)

# t = 0
s_init = lambda x,y: cos(x*y) # t = 0
i_init = lambda x,y: exp(x)*sin(y) # t = 0

f = lambda x,y,t: cos(x*y)*exp(t)*(1+exp(x)*sin(y)*cos(t)+y**2+x**2)
g = lambda x,y,t: -exp(x)*sin(y)*(sin(t)+cos(t)*cos(x*y)*exp(t))

xa = 0.0; xb = 1.0; ya = 0.0; yb = 1.0


s_xa = lambda x,y,t: s_exact(x,ya,t)
s_xb = lambda x,y,t: s_exact(x,yb,t)
s_ya = lambda x,y,t: s_exact(xa,y,t)
s_yb = lambda x,y,t: s_exact(xb,y,t)

i_xa = lambda x,y,t: i_exact(x,ya,t)
i_xb = lambda x,y,t: i_exact(x,yb,t)
i_ya = lambda x,y,t: i_exact(xa,y,t)
i_yb = lambda x,y,t: i_exact(xb,y,t)

# Parameters
lambda_si = 0;

def solve(h,dt):
    
    iter = int(1/dt)
    t = 0
    l = dt/(h**2)
    
    nx=int((xb - xa)/h)-1;
    ny=int((yb - ya)/h)-1;

    # Memory allocation

    x=np.linspace(xa,xb,nx+2)
    y=np.linspace(ya,yb,ny+2)

    s0 = zeros([nx+2,ny+2],float)
    i0 = zeros([nx+2,ny+2],float)
    s1 = zeros([nx+2,ny+2],float)
    i1 = zeros([nx+2,ny+2],float)
    s_e = zeros([nx+2,ny+2],float)
    i_e = zeros([nx+2,ny+2],float)
    rhs_s = zeros([nx+2,ny+2],float)
    rhs_i = zeros([nx+2,ny+2],float)
    res_s = zeros([nx+2,ny+2],float)
    res_i = zeros([nx+2,ny+2],float)




    for i in range (nx+2):
        for j in range (ny+2):   
                
            s0[i,j] = s_init(x[i],y[j])
            i0[i,j] = i_init(x[i],y[j])
            
    # Exact solution

    for i in range (nx+2):
        for j in range (ny+2):
            s_e[i,j] = s_exact(x[i],y[j],1)
            i_e[i,j] = i_exact(x[i],y[j],1)

        
    for k in range(iter):
        t += dt

                
        # Boundary Condition

        for i in range(nx+2):
             s1[i,0] = s_xa(x[i],y[0],t)
             s1[i,ny+1] = s_xb(x[i],y[ny+1],t)
            
             i1[i,0] = i_xa(x[i],y[0],t)
             i1[i,ny+1] = i_xb(x[i],y[ny+1],t)
            
            
            
        for j in range(ny+2):
            s1[0,j] = s_ya(x[0],y[j],t)
            s1[nx+1,j] = s_yb(x[nx+1],y[j],t)

            i1[0,j] = i_ya(x[0],y[j],t)
            i1[nx+1,j] = i_yb(x[nx+1],y[j],t)
            

        # ------------- calculation for S ---------------

        # Calculate RHS
            
        for i in range (1,nx+1):
            for j in range (1,nx+1):
                            
                rhs_s[i,j] = s0[i,j] + f(x[i],y[j],t) * dt
                
                    
        ratio = 1.0; r=0
        
        while(ratio > 1e-7):
            
            r=r+1
                                     
            # Calculate Residue
            
            for i in range (1,nx+1):
                for j in range (1,ny+1):
                    
                    alpha =  1 + 4*l + i0[i,j] * dt 
                    res_s[i,j] = rhs_s[i,j] + l*(s1[i-1,j]+ s1[i+1,j] + s1[i,j+1]+ s1[i,j-1]) - alpha * s1[i,j]
                                   

            rn=(max(abs(res_s)));
            if r==1:r0=rn
            ratio = rn/r0
                            
            # Update the solution
            
            for i in range (1,nx+1):
                for j in range (1,ny+1):
                    
                    alpha = 1 + 4*l + i0[i,j] * dt
                    
                    s1[i,j] = (1/alpha)*(rhs_s [i,j] + l*(s1[i-1,j] + s1[i+1,j] + s1[i,j+1] + s1[i,j-1]))

        s0 = s1.copy()
      

        # ------------- calculation for I ---------------  

                    
        # Calculate RHS
            
        for i in range (1,nx+1):
            for j in range (1,ny+1):
                                    
                rhs_i[i,j] = i0[i,j] + g(x[i],y[j],t) * dt

                
        ratio = 1.0; r=0
        
        while(ratio > 1e-7):
            
            r=r+1

                    

            # Calculate Residue
            
            for i in range (1,nx+1):
               for j in range (1,ny+1):

                    beta = 1 + 4*l + (lambda_si - s0[i,j]) * dt
                    
                    res_i[i,j] = rhs_i [i,j]+ l*(i1[i-1,j]+ i1[i+1,j] + i1[i,j+1]+ i1[i,j-1]) - beta * i1[i,j]

            rn=(max(abs(res_i)));
            if r==1:r0=rn
            ratio = rn/r0
                    
            # Update the solution
            
            for i in range (1,nx+1):
                for j in range (1,ny+1):
                    
                    beta = 1 + 4*l + (lambda_si - s0[i,j]) * dt
                    
                    i1[i,j] = (1/beta)*(rhs_i [i,j]+ l*(i1[i-1,j]+ i1[i+1,j] + i1[i,j+1]+ i1[i,j-1]))

        i0 = i1.copy()
        print(t,dt,h,r,"\n")     
                         
    rms_s = np.sqrt(np.mean((s0-s_e)**2))
    rms_i = np.sqrt(np.mean((i0-i_e)**2))
    return rms_s,rms_i

h_arr = [1/4,1/8,1/16,1/32]
dt_arr = [1/2,1/4,1/8,1/16,1/32,1/64]

h_result = np.zeros((len(h_arr),5))
i = 0
for h in h_arr:
    start = timer()
    h_result[i,0], h_result[i,1] = solve(h,1e-4)
    end = timer()
    h_result[i,4] = end-start
    i += 1
for i in range(len(h_arr)-1):
    h_result[i+1,2] = h_result[i,0]/h_result[i+1,0]
    h_result[i+1,3] = h_result[i,1]/h_result[i+1,1]
        
for i in range(len(h_arr)):
    print('h = {}, dt = {}'.format(h_arr[i], 1e-4))
    print('{:.8e} {:.8e} {:.8e} {:.8e} {:.4f}'.format(h_result[i,0],h_result[i,1],h_result[i,2],h_result[i,3],h_result[i,4]))

    
print('\n\n')
dt_result = np.zeros((len(dt_arr),5))
i = 0
for dt in dt_arr:
    start = timer()
    dt_result[i,0], dt_result[i,1] = solve(1/64,dt)
    end = timer()
    dt_result[i,4] = end - start
    i += 1

for i in range(len(dt_arr)-1):
    dt_result[i+1,2] = dt_result[i,0]/dt_result[i+1,0]
    dt_result[i+1,3] = dt_result[i,1]/dt_result[i+1,1]
        
for i in range(len(dt_arr)):
    print('h = {}, dt = {}'.format(1/64, dt_arr[i]))
    print('{:.8e} {:.8e} {:.8e} {:.8e} {:.4f}'.format(dt_result[i,0],dt_result[i,1],dt_result[i,2],dt_result[i,3],dt_result[i,4])) 
             
        
