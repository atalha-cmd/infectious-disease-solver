
import numpy as np
from numpy import cos,sin,exp
import matplotlib.pyplot as plt    
from timeit import default_timer as timer
import sys
import warnings
from numpy.lib.npyio import save
warnings.filterwarnings('ignore')
# initial values, boundary values, sources, true solutions
s_exact = lambda x,y,t: cos(x*y)*exp(t)
i_exact = lambda x,y,t: exp(x)*sin(y)*cos(t)

# t = 0
s_init = lambda x,y: cos(x*y) # t = 0
i_init = lambda x,y,dt: exp(x)*sin(y)*cos(-0.5*dt) # t = -1/2

# x [0,1] ; y[0,1]
s_xa = lambda x,t: exp(t)
s_xb = lambda x,t: cos(x)*exp(t)
s_ya = lambda y,t: exp(t)
s_yb = lambda y,t: cos(y)*exp(t)
i_xa = lambda x,t: 0
i_xb = lambda x,t: exp(x)*sin(1)*cos(t)
i_ya = lambda y,t: sin(y)*cos(t)
i_yb = lambda y,t: exp(1)*sin(y)*cos(t)

lambda_si = 0
f = lambda x,y,t: cos(x*y)*exp(t)*(1+exp(x)*sin(y)*cos(t)+y**2+x**2)
g = lambda x,y,t: -exp(x)*sin(y)*(sin(t)+cos(t)*cos(x*y)*exp(t))

# construct matrix A with diagonal bands 
def buildMatrixA(N,n,l):
    I = np.identity(N)
    A = (1+4*l)*I
    c = 0
    for i in range(N):
        if c == n-1:
            c = 0
            continue
        A[i, i+1] = -l
        A[i+1, i] = -l
        c += 1
    
    for i in range(N-n):
        A[i, n+i] = -l
        A[n+i, i] = -l
    return A

def generateArrays(n,N,x,y,dt):
    xy_pair = np.zeros((N,2))
    s0 = np.zeros(N)
    i0 = np.zeros(N)
    s_e = np.zeros(N)
    i_e = np.zeros(N)
    c = 0
    for i in x:
        for j in y:
            xy_pair[c,0] = i
            xy_pair[c,1] = j
            s0[c] = s_init(i,j)
            i0[c] = i_init(i,j,dt)
            s_e[c] = s_exact(i,j,1)
            i_e[c] = i_exact(i,j,1) # for staggered, i needs to shift back half of the dt
            c += 1
    return xy_pair, s0, i0, s_e, i_e

# main function for solving the PDE problem
def solve(h, dt):
    # -------------initialize parameters-------------
    x = np.arange(h, 1, h)
    y = np.arange(h, 1, h)
    n = len(x)
    N = n*n
    iter = int(1/dt)
    t = 0
    l = dt/(2*h**2)

    # -------------initialize arrays-------------
    A = buildMatrixA(N,n,l)
    As = np.copy(A); Ai = np.copy(A)
    s1 = np.zeros(N); i1 = np.zeros(N)
    rhs_s = np.zeros(N); rhs_i = np.zeros(N)

    # xy_pair is used for conversion between 1d array (when we solve the lin alg system, the solution is 1d)and 2d grid given (x,y coordinates)
    xy_pair, s0, i0, s_e, i_e = generateArrays(n,N,x,y,dt)
    
    # -----loop for time step (dt)-----
    for k in range(iter):
        t += dt #increment dt 

        # ------------- calculation for I ---------------
        for m in range(N):
            # update diagonal coefficient
            Ai[m,m] = A[m,m] + (lambda_si-s0[m])*dt*.5
            # construct right hand side
            rhs_i[m] = (-(lambda_si-s0[m])*dt*0.5+1-4*l)*i0[m] \
                        + (g(xy_pair[m,0],xy_pair[m,1],t-0.5*dt)+g(xy_pair[m,0],xy_pair[m,1],t-0.5*dt-dt))*dt*.5

            # update points on the boundary in x,y coordinates
            if m < N-n: # every point has a right neighbor; excludes the right-most column
                rhs_i[m] += l*i0[m+n]
            if m > n-1: # every point has a left neighbor; excludes the left-most column
                rhs_i[m] += l*i0[m-n]
            if m % n != 0: # every point has a upper neighbor; excludes the top row
                rhs_i[m] += l*i0[m-1]
            if m % n != n-1: # every point has a lower neighbor; excludes the bottom row
                rhs_i[m] += l*i0[m+1]

        # update boundary points
        for m in range(n):
            rhs_i[m] += l*i_ya(y[m], t-0.5*dt) + l*i_ya(y[m], t-0.5*dt-dt)
            rhs_i[m*n] += l*i_xa(x[m], t-0.5*dt) + l*i_xa(x[m], t-0.5*dt-dt)
            rhs_i[(m+1)*n-1] += l*i_xb(x[m], t-0.5*dt) + l*i_xb(x[m], t-0.5*dt-dt)
            rhs_i[-n+m] += l*i_yb(y[m], t-0.5*dt) + l*i_yb(y[m], t-0.5*dt-dt)

        i1 = np.linalg.solve(Ai, rhs_i)
        i0 = i1.copy()
        
        # ------------- calculation for S ---------------
        for m in range(N):
            # update diagonal coefficient
            As[m,m] = A[m,m] + i1[m]*dt*.5
            # construct right hand side
            rhs_s[m] = (-i1[m]*dt*0.5+1-4*l)*s0[m] \
                        + (f(xy_pair[m,0],xy_pair[m,1],t)+f(xy_pair[m,0],xy_pair[m,1],t-dt))*dt*.5
            
            # update points on the boundary in x,y coordinates
            if m < N-n:  # every point has a right neighbor; excludes the right-most column
                rhs_s[m] += l*s0[m+n]
            if m > n-1: # every point has a left neighbor; excludes the left-most column
                rhs_s[m] += l*s0[m-n]
            if m % n != 0: # every point has a upper neighbor; excludes the top row
                rhs_s[m] += l*s0[m-1]
            if m % n != n-1: # every point has a lower neighbor; excludes the bottom row
                rhs_s[m] += l*s0[m+1]

        # update boundary points
        for m in range(n):
            rhs_s[m] += l*s_ya(y[m], t) + l*s_ya(y[m], t-dt)
            rhs_s[m*n] += l*s_xa(x[m], t) + l*s_xa(x[m], t-dt)
            rhs_s[(m+1)*n-1] += l*s_xb(x[m], t) + l*s_xb(x[m], t-dt)
            rhs_s[-n+m] += l*s_yb(y[m], t) + l*s_yb(y[m], t-dt)


        s1 = np.linalg.solve(As, rhs_s)
        s0 = s1.copy()
        print(t,dt,h,"\n\n")

    # handle extra 1/2 step of I
    t += dt #increment dt 

    # ------------- calculation for I ---------------
    for m in range(N):
        # update diagonal coefficient
        Ai[m,m] = A[m,m] + (lambda_si-s0[m])*dt*.5
        # construct right hand side
        rhs_i[m] = (-(lambda_si-s0[m])*dt*0.5+1-4*l)*i0[m] \
                    + (g(xy_pair[m,0],xy_pair[m,1],t-0.5*dt)+g(xy_pair[m,0],xy_pair[m,1],t-0.5*dt-dt))*dt*.5

        # update points on the boundary in x,y coordinates
        if m < N-n: # every point has a right neighbor; excludes the right-most column
            rhs_i[m] += l*i0[m+n]
        if m > n-1: # every point has a left neighbor; excludes the left-most column
            rhs_i[m] += l*i0[m-n]
        if m % n != 0: # every point has a upper neighbor; excludes the top row
            rhs_i[m] += l*i0[m-1]
        if m % n != n-1: # every point has a lower neighbor; excludes the bottom row
            rhs_i[m] += l*i0[m+1]

    # update boundary points
    for m in range(n):
        rhs_i[m] += l*i_ya(y[m], t-0.5*dt) + l*i_ya(y[m], t-0.5*dt-dt)
        rhs_i[m*n] += l*i_xa(x[m], t-0.5*dt) + l*i_xa(x[m], t-0.5*dt-dt)
        rhs_i[(m+1)*n-1] += l*i_xb(x[m], t-0.5*dt) + l*i_xb(x[m], t-0.5*dt-dt)
        rhs_i[-n+m] += l*i_yb(y[m], t-0.5*dt) + l*i_yb(y[m], t-0.5*dt-dt)

    i1 = np.linalg.solve(Ai, rhs_i)

    # calculate root mean square error
    rms1 = np.sqrt(np.mean((s1-s_e)**2))
    rms2 = np.sqrt(np.mean(((i1+i0)/2-i_e)**2))
    
    return rms1,rms2




if __name__ == "__main__":

    save_file = open('Solver_result.csv','w')
    h_arr = [1/4,1/8,1/16,1/32,1/64]
    dt_arr = [1/2,1/4,1/8,1/16,1/32,1/64]


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
        







