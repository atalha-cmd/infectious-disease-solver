
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

    alpha_i = zeros([nx+2,ny+2],float)
    alpha_s = zeros([nx+2,ny+2],float)
    beta_i = zeros([nx+2,ny+2],float)
    beta_s = zeros([nx+2,ny+2],float)
    s0 = zeros([nx+2,ny+2],float)
    i0 = zeros([nx+2,ny+2],float)
    s_p = zeros([nx+2,ny+2],float)
    i_p = zeros([nx+2,ny+2],float)
    s1 = zeros([nx+2,ny+2],float)
    i1 = zeros([nx+2,ny+2],float)
    s_e = zeros([nx+2,ny+2],float)
    i_e = zeros([nx+2,ny+2],float)


    # LU factorization based on Thomas's algorithm
    # input: a, b, c, r - matrix elements and right hand side vector
    # output: w - solution of linear system

    def LU_Thomas(a,b,c,r):
        
        n = len(r)
        w = zeros(n,float)
        l = zeros(n,float)
        u = zeros(n,float)
        z = zeros(n,float)
        #Determine L,U factors
        u[0] = b[0]
        for k in range(1,n):
            l[k] = a[k]/u[k-1]
            u[k] = b[k]-l[k]*c[k-1]
        # Solve Lz = r
        z[0] = r[0]
        for k in range(1,n):
            z[k] = r[k]-l[k]*z[k-1]

        # Solve Uw = z.
        w[n-1] = z[n-1]/u[n-1]
        for k in range(n-2,-1,-1):
            w[k] = (z[k]-c[k]*w[k+1])/u[k]

        return w

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


    # Implement Peaceman-Rachford ADI scheme

    for k in range(iter):
        t += dt
        # Boundary Condition
        for i in range(nx+2):
            s1[i,0] = s_xa(x[i],y[0],t)
            s1[i,ny+1] = s_xb(x[i],y[ny+1],t)

            s_p[i,0] = s_xa(x[i],y[0],t - 0.5*dt)
            s_p[i,ny+1] = s_xb(x[i],y[ny+1],t - 0.5*dt)
        
            i1[i,0] = i_xa(x[i],y[0],t - 0.5*dt)
            i1[i,ny+1] = i_xb(x[i],y[ny+1],t - 0.5*dt)

            i_p[i,0] = i_xa(x[i],y[0],t - dt)
            i_p[i,ny+1] = i_xb(x[i],y[ny+1],t - dt)
            
            
        for j in range(ny+2):
            s1[0,j] = s_ya(x[0],y[j],t)
            s1[nx+1,j] = s_yb(x[nx+1],y[j],t)

            s_p[0,j] = s_ya(x[0],y[j],t - 0.5*dt)
            s_p[nx+1,j] = s_yb(x[nx+1],y[j],t - 0.5*dt)

            i1[0,j] = i_ya(x[0],y[j],t - 0.5*dt)
            i1[nx+1,j] = i_yb(x[nx+1],y[j],t - 0.5*dt)

            i_p[0,j] = i_ya(x[0],y[j],t - dt)
            i_p[nx+1,j] = i_yb(x[nx+1],y[j],t - dt)
        
        # ------------- calculation for I --------------- 

        # Calculate the Co-Eficients
        
        for j in range(1,ny+1):
            for i in range(1,nx+1):
                alpha_i[i,j] = 1 + (dt/4)*(lambda_si - s0[i,j])
                beta_i[i,j]  = 1 - (dt/4)*(lambda_si - s0[i,j])


        # Step 1: Solve the x-direction problem using the Thomas algorithm
    
        for j in range(1,ny+1):
            a_x = zeros(nx,float)
            b_x = zeros(nx,float)
            c_x = zeros(nx,float)
            r_x = zeros(nx,float)

            for i in range(nx):
                a_x[i] = -l
                b_x[i] = alpha_i[i+1,j] + 2*l
                c_x[i] = -l

                r_x[i] = l * i0[i+1,j-1] + (beta_i[i+1,j] - 2*l) * i0[i+1,j] + l * i0[i+1,j+1] \
                                                + 0.5*dt*(g(x[i+1],y[j],t-dt)) 
                        
            r_x[0] += l*i_p[0,j]      
            r_x[-1] +=  l*i_p[nx+1,j]
            
            i_p[1:nx+1,j] = LU_Thomas(a_x, b_x, c_x, r_x)
            
        # Step 2: Solve the y-direction problem using the Thomas algorithm
        for i in range(1,nx+1):
            a_y = zeros(ny,float)
            b_y = zeros(ny,float)
            c_y = zeros(ny,float)
            r_y = zeros(ny,float)

            for j in range(ny):
                a_y[j] = -l
                b_y[j] = alpha_i[i,j+1] + 2*l
                c_y[j] = -l

                r_y[j] = l * i_p[i-1,j+1] + (beta_i[i,j+1] - 2*l) * i_p[i,j+1] + l * i_p[i+1,j+1] \
                                            + 0.5*dt*(g(x[i],y[j+1],t-dt)) 

            r_y[0]  += l*i1[i,0]          
            r_y[-1] += l*i1[i,ny+1]

            i1[i,1:ny+1] = LU_Thomas(a_y, b_y, c_y, r_y)
                  
        i0 = i1.copy()


        # ------------- calculation for S --------------- 

        # Calculate the Co-Eficients
        
        for j in range(1,ny+1):
            for i in range(1,nx+1):
                alpha_s[i,j] = 1 + (dt/4)*i0[i,j]
                beta_s[i,j]  = 1 - (dt/4)*i0[i,j]

        # Step 1: Solve the x-direction problem using the Thomas algorithm
        for j in range(1,ny+1):
            a_x = zeros(nx,float)
            b_x = zeros(nx,float)
            c_x = zeros(nx,float)
            r_x = zeros(nx,float)
            
            for i in range(nx):      
                a_x[i] = -l
                b_x[i] = alpha_s[i+1,j] + 2*l
                c_x[i] = -l

                r_x[i] = l * s0[i+1,j-1] + (beta_s[i+1,j] - 2*l) * s0[i+1,j] + l * s0[i+1,j+1] \
                                            + 0.5*dt*(f(x[i+1],y[j],t-0.5*dt))
    
            r_x[0] += l*s_p[0,j]         
            r_x[-1] += l*s_p[nx+1,j]

            s_p[1:nx+1,j] = LU_Thomas(a_x, b_x, c_x, r_x)
            
        # Step 2: Solve the y-direction problem using the Thomas algorithm
        for i in range(1,nx+1):
            a_y = zeros(ny,float)
            b_y = zeros(ny,float)
            c_y = zeros(ny,float)
            r_y = zeros(ny,float)

            for j in range(ny):
                a_y[j] = -l
                b_y[j] = alpha_s[i,j+1] + 2*l
                c_y[j] = -l

                r_y[j] = l * s_p[i-1,j+1] + (beta_s[i,j+1] - 2*l) * s_p[i,j+1] + l * s_p[i+1,j+1] \
                                            + 0.5*dt*(f(x[i],y[j+1],t-0.5*dt))
                
            r_y[0] += l*s1[i,0]         
            r_y[-1] += l*s1[i,ny+1]

            s1[i,1:ny+1] = LU_Thomas(a_y, b_y, c_y, r_y)
    
        s0 = s1.copy()

        print(t,dt,h,"\n\n")
    
    # # # handle extra 1/2 step of I

    # ------------- calculation for I --------------- 

    t += dt #increment dt 

    # Boundary Condition
    for i in range(nx+2):
    
        i1[i,0] = i_xa(x[i],y[0],t - 0.5*dt)
        i1[i,ny+1] = i_xb(x[i],y[ny+1],t - 0.5*dt)

        i_p[i,0] = i_xa(x[i],y[0],t - dt)
        i_p[i,ny+1] = i_xb(x[i],y[ny+1],t - dt)
        
        
    for j in range(ny+2):
 
        i1[0,j] = i_ya(x[0],y[j],t - 0.5*dt)
        i1[nx+1,j] = i_yb(x[nx+1],y[j],t - 0.5*dt)

        i_p[0,j] = i_ya(x[0],y[j],t - dt)
        i_p[nx+1,j] = i_yb(x[nx+1],y[j],t - dt)
    


    # Step 1: Solve the x-direction problem using the Thomas algorithm

    # Calculate the Co-Eficients

    for j in range(1,ny+1):
            for i in range(1,nx+1):
                alpha_i[i,j] = 1 + (dt/4)*(lambda_si - s0[i,j])
                beta_i[i,j]  = 1 - (dt/4)*(lambda_si - s0[i,j])
        
    for j in range(1,ny+1):
        a_x = zeros(nx,float)
        b_x = zeros(nx,float)
        c_x = zeros(nx,float)
        r_x = zeros(nx,float)
        
        for i in range(nx):
            a_x[i] = -l
            b_x[i] = alpha_i[i+1,j] + 2*l
            c_x[i] = -l

            r_x[i] = l * i0[i+1,j-1] + (beta_i[i+1,j] - 2*l) * i0[i+1,j] + l * i0[i+1,j+1] \
                                        + 0.5*dt*(g(x[i+1],y[j],t-dt))

        r_x[0]  += l*i_p[0,j]      
        r_x[-1] += l*i_p[nx+1,j]
        
        i_p[1:nx+1,j] = LU_Thomas(a_x, b_x, c_x, r_x)
        
    # Step 2: Solve the y-direction problem using the Thomas algorithm

    for i in range(1,nx+1):
        a_y = zeros(ny,float)
        b_y = zeros(ny,float)
        c_y = zeros(ny,float)
        r_y = zeros(ny,float)

        for j in range(ny):
            a_y[j] = -l
            b_y[j] = alpha_i[i,j+1] + 2*l
            c_y[j] = -l

            r_y[j] = l * i_p[i-1,j+1] + (beta_i[i,j+1] - 2*l) * i_p[i,j+1] + l * i_p[i+1,j+1] \
                                        + 0.5*dt*(g(x[i],y[j+1],t-dt))

        r_y[0]  += l*i1[i,0]      
        r_y[-1] += l*i1[i,ny+1]

        i1[i,1:ny+1] = LU_Thomas(a_y, b_y, c_y, r_y)
      
    rms_s = np.sqrt(np.mean((s0 - s_e)**2))
    rms_i = np.sqrt(np.mean(((i0+i1)*0.5 -i_e)**2))

    return rms_s,rms_i 


if __name__ == "__main__":

    save_file = open('ADI_result.csv','w')
    h_arr = [1/2,1/4,1/8,1/16,1/32,1/64]
    dt_arr = [1/2,1/4,1/8,1/16,1/32,1/64]


    # # running cases where h is changing, dt is fixed
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



    # running cases where dt is changing, h is fixed
    print('\n\n')
    dt_result = np.zeros((len(dt_arr),5))
    i = 0


    save_file.write('\n')
    h = 1/64
    for dt in dt_arr:
        start = timer()
        dt_result[i,0], dt_result[i,1] = solve(h,dt)
        end = timer()
        dt_result[i,4] = end - start
        i += 1

    for i in range(len(dt_arr)-1):
        dt_result[i+1,2] = dt_result[i,0]/dt_result[i+1,0]
        dt_result[i+1,3] = dt_result[i,1]/dt_result[i+1,1]

    # ----------------- writing outputs ----------------
    print('======================================================================================================================================================')
    print('| {:^31} | {:^14s} | {:^14s} | {:^14s} | {:^14s} | {:^14s} | {:^14s} | {:^11s} |'.format('h    dt','rms of S', 'rms of I', 'times of S', 'times of I', 'order of S', 'order of I', 'seconds'))
    print('======================================================================================================================================================')
    save_file.write('Fix h and changing dt\n')
    save_file.write('h,dt,rms of S,rms of I,times of S,times of I,order of S,order of I,seconds\n')
    for i in range(len(dt_arr)):
        # print('h = {}, dt = {}'.format(h, dt_arr[i]))
        # print('{:.8e} {:.8e} {:.8e} {:.8e} {:.4f}'.format(dt_result[i,0],dt_result[i,1],dt_result[i,2],dt_result[i,3],dt_result[i,4])) 
        print('| h = {:>10}, dt = {:>10} '.format(h, dt_arr[i]), end='')
        print('| {:>14.8e} | {:>14.8e} | {:>14.8e} | {:>14.8e} | {:>14.8e} | {:>14.8e} | {:>10.4f}s |'.format(dt_result[i,0],dt_result[i,1],dt_result[i,2],dt_result[i,3],np.log2(dt_result[i,2]),np.log2(dt_result[i,3]),dt_result[i,4]))
        print('------------------------------------------------------------------------------------------------------------------------------------------------------')
        save_file.write(str(h)+','+str(dt_arr[i])+','+str(dt_result[i,0])+','+str(dt_result[i,1])+','+str(dt_result[i,2])+','+str(dt_result[i,3])+','+str(np.log2(dt_result[i,2]))+','+str(np.log2(dt_result[i,3]))+','+str(dt_result[i,4])+'\n')
        
    save_file.close()
   