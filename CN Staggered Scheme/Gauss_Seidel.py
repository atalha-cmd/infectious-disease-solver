# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:15:25 2023

@author: TALHA
"""

import numpy as np
from numpy import cos,sin,exp,zeros,max
from math import exp,log,sqrt,cos,pi,sin
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import sys
import warnings
warnings.filterwarnings('ignore')

s_exact = lambda x,y,t: cos(x*y)*exp(t)
i_exact = lambda x,y,t: exp(x)*sin(y)*cos(t)

# t = 0
s_init = lambda x,y: cos(x*y) # t = 0
i_init = lambda x,y,dt: exp(x)*sin(y)*cos(-0.5*dt) # t = -1/2

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

lambda_si = 0.0;

def solve(h,dt):

    iter = int(1/dt)
    t = 0
    l = dt/(2*h**2) 
 
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

    # Initial Solution

    for i in range (nx+2):
        for j in range (ny+2):   

            s0[i,j] = s_init(x[i],y[j])
            i0[i,j] = i_init(x[i],y[j],dt)
    
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
            
             i1[i,0] = i_xa(x[i],y[0],t - 0.5*dt)
             i1[i,ny+1] = i_xb(x[i],y[ny+1],t - 0.5*dt)
            
            
            
        for j in range(ny+2):
            s1[0,j] = s_ya(x[0],y[j],t)
            s1[nx+1,j] = s_yb(x[nx+1],y[j],t)

            i1[0,j] = i_ya(x[0],y[j],t - 0.5*dt)
            i1[nx+1,j] = i_yb(x[nx+1],y[j],t - 0.5*dt)

          
       
        # ------------- calculation for I ---------------  
        # Calculate RHS
            
        for i in range (1,nx+1):
            for j in range (1,ny+1):

                rhs_i[i,j] = (((s0[i,j]-lambda_si)*0.5*dt) + 1 - 4*l)*i0[i,j] + l*(i0[i-1,j] + i0[i+1,j] + i0[i,j+1] + i0[i,j-1]) \
                                  + 0.5*dt*(g(x[i],y[j],t - 0.5*dt) + g(x[i],y[j],t - 1.5*dt))


        ratio = 1.0; r=0
        while(ratio > 1e-7):
            r=r+1

            # Calculate Residue
            
            for i in range (1,nx+1):
               for j in range (1,ny+1):
                   
                    beta = 1 + 4*l + 0.5*dt*(lambda_si - s0[i,j])        
                    res_i[i,j] = rhs_i [i,j]+ l*(i1[i-1,j]+ i1[i+1,j] + i1[i,j+1]+ i1[i,j-1]) - beta * i1[i,j]


            rn=(max(abs(res_i)));
            if r==1:r0=rn
            ratio = rn/r0
            
                    
            # Update the solution
            
            for i in range (1,nx+1):
                for j in range (1,ny+1):
                    beta = 1 + 4*l + 0.5*dt*(lambda_si - s0[i,j])
                    i1[i,j] = (1/beta)*(rhs_i [i,j]+ l*(i1[i-1,j]+ i1[i+1,j] + i1[i,j+1]+ i1[i,j-1]))


        i0 = i1.copy()
                    

             
        # ------------- calculation for S ---------------    
        # Calculate RHS
            
        for i in range (1,nx+1):
            for j in range (1,ny+1):

                rhs_s[i,j] = ((-0.5*dt)*i0[i,j]+ 1 - 4*l)*s0[i,j] + l*(s0[i-1,j] + s0[i+1,j] + s0[i,j+1]+ s0[i,j-1]) \
                                 + 0.5*dt*(f(x[i],y[j],t) + f(x[i],y[j],t-dt))

                    
        ratio = 1.0; r=0
        while(ratio > 1e-7):
            
            r=r+1
                       
            # Calculate Residue
            
            for i in range (1,nx+1):
                for j in range (1,ny+1):

                    alpha = 1 + 4*l + 0.5*dt*i0[i,j]
                    res_s[i,j] = rhs_s[i,j] + l*(s1[i-1,j]+ s1[i+1,j] + s1[i,j+1]+ s1[i,j-1]) - alpha*s1[i,j]
          
                                   

            rn=(max(abs(res_s)));
            if r==1:r0=rn
            ratio = rn/r0
                            
            # Update the solution
            
            for i in range (1,nx+1):
                for j in range (1,ny+1):

                    alpha = 1 + 4*l + 0.5*dt*i0[i,j]
                    s1[i,j] = (1/alpha)*(rhs_s [i,j] + l*(s1[i-1,j]+ s1[i+1,j] + s1[i,j+1]+ s1[i,j-1]))

        s0 = s1.copy()
        print(t,dt,h, r,"\n\n")


    # # # handle extra 1/2 step of I

    t += dt #increment dt
      
            
    # ------------- calculation for I --------------- 
     # Boundary Condition

    for i in range(nx+2):
        s1[i,0] = s_xa(x[i],y[0],t)
        s1[i,ny+1] = s_xb(x[i],y[ny+1],t)
    
        i1[i,0] = i_xa(x[i],y[0],t - 0.5*dt)
        i1[i,ny+1] = i_xb(x[i],y[ny+1],t - 0.5*dt)
        
        
        
    for j in range(ny+2):
        s1[0,j] = s_ya(x[0],y[j],t)
        s1[nx+1,j] = s_yb(x[nx+1],y[j],t)

        i1[0,j] = i_ya(x[0],y[j],t - 0.5*dt)
        i1[nx+1,j] = i_yb(x[nx+1],y[j],t - 0.5*dt)




    # Calculate RHS     
    for i in range (1,nx+1):
        for j in range (1,ny+1):

            rhs_i[i,j] = (((s0[i,j]-lambda_si)*0.5*dt) + 1 - 4*l)*i0[i,j] + l*(i0[i-1,j] + i0[i+1,j] + i0[i,j+1] + i0[i,j-1]) \
                                + 0.5*dt*(g(x[i],y[j],t - 0.5*dt) + g(x[i],y[j],t - 1.5*dt))


    ratio = 1.0; r=0
    while(ratio > 1e-7):
        r=r+1

        # Calculate Residue
        
        for i in range (1,nx+1):
            for j in range (1,ny+1):
                
                beta = 1 + 4*l + 0.5*dt*(lambda_si - s0[i,j])        
                res_i[i,j] = rhs_i [i,j]+ l*(i1[i-1,j]+ i1[i+1,j] + i1[i,j+1]+ i1[i,j-1]) - beta * i1[i,j]

        rn=(max(abs(res_i)));
        if r==1:r0=rn
        ratio = rn/r0  
                
        # Update the solution
        
        for i in range (1,nx+1):
            for j in range (1,ny+1):
                beta = 1 + 4*l + 0.5*dt*(lambda_si - s0[i,j])
                i1[i,j] = (1/beta)*(rhs_i [i,j]+ l*(i1[i-1,j]+ i1[i+1,j] + i1[i,j+1]+ i1[i,j-1]))


    rms_s = np.sqrt(np.mean((s0 - s_e)**2))
    rms_i = np.sqrt(np.mean(((i0+i1)*0.5 -i_e)**2))

    return rms_s,rms_i  


if __name__ == "__main__":

    save_file = open('Gauss_result.csv','w')
    h_arr = [1/4,1/8,1/16,1/32,1/64]


    # running cases where h is changing, dt is fixed
    h_result = np.zeros((len(h_arr),5))
    i = 0; dt = 1e-4
    for h in h_arr:
        start = timer()
        h_result[i,0], h_result[i,1] = solve(h,dt)
        end = timer()
        h_result[i,4] = end-start
        i += 1

    for i in range(len(h_arr)-1):
        h_result[i+1,2] = h_result[i,0]/h_result[i+1,0]
        h_result[i+1,3] = h_result[i,1]/h_result[i+1,1]

    # ----------------- writing outputs ----------------
    save_file.write('Changing h and fix dt\n')
    save_file.write('h,dt,rms of S,rms of I,times of S,times of I,order of S,order of I,seconds\n')
    print('======================================================================================================================================================')
    print('| {:^31} | {:^14s} | {:^14s} | {:^14s} | {:^14s} | {:^14s} | {:^14s} | {:^11s} |'.format('h    dt','rms of S', 'rms of I', 'times of S', 'times of I', 'order of S', 'order of I', 'seconds'))
    print('======================================================================================================================================================')
    for i in range(len(h_arr)):
        print('| h = {:>10}, dt = {:>10} '.format(h_arr[i], dt), end='')
        print('| {:>14.8e} | {:>14.8e} | {:>14.8e} | {:>14.8e} | {:>14.8e} | {:>14.8e} | {:>10.4f}s |'.format(h_result[i,0],h_result[i,1],h_result[i,2],h_result[i,3],np.log2(h_result[i,2]),np.log2(h_result[i,3]),h_result[i,4]))
        print('------------------------------------------------------------------------------------------------------------------------------------------------------')
        save_file.write(str(h_arr[i])+','+str(dt)+','+str(h_result[i,0])+','+str(h_result[i,1])+','+str(h_result[i,2])+','+str(h_result[i,3])+','+str(np.log2(h_result[i,2]))+','+str(np.log2(h_result[i,3]))+','+str(h_result[i,4])+'\n')



    save_file.close()
   