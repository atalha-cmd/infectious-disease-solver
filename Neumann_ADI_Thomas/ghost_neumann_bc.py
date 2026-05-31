import numpy as np
from numpy import zeros
from math import exp, sin, cos
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Exact solutions
# ============================================================
s_exact = lambda x, y, t: cos(x*y) * exp(t)
i_exact = lambda x, y, t: exp(x) * sin(y) * cos(t)

# Initial conditions
s_init = lambda x, y: cos(x*y)                                # S at t=0
i_init = lambda x, y, dt: exp(x) * sin(y) * cos(-0.5*dt)      # I at t=-dt/2

# Source terms
f = lambda x, y, t: cos(x*y) * exp(t) * (1 + exp(x)*sin(y)*cos(t) + y**2 + x**2)
g = lambda x, y, t: -exp(x) * sin(y) * (sin(t) + cos(t)*cos(x*y)*exp(t))

# Domain
xa, xb = 0.0, 1.0
ya, yb = 0.0, 1.0

# Derivatives for Neumann flux q = du/dn
Sx = lambda x, y, t: -y * sin(x*y) * exp(t)
Sy = lambda x, y, t: -x * sin(x*y) * exp(t)
Ix = lambda x, y, t: exp(x) * sin(y) * cos(t)
Iy = lambda x, y, t: exp(x) * cos(y) * cos(t)
# Sx = lambda x, y, t: 0
# Sy = lambda x, y, t: 0
# Ix = lambda x, y, t: 0
# Iy = lambda x, y, t: 0

# outward normal flux q = du/dn
S_L = lambda y, t: -Sx(xa, y, t)
S_R = lambda y, t:  Sx(xb, y, t)
S_B = lambda x, t: -Sy(x, ya, t)
S_T = lambda x, t:  Sy(x, yb, t)

I_L = lambda y, t: -Ix(xa, y, t)
I_R = lambda y, t:  Ix(xb, y, t)
I_B = lambda x, t: -Iy(x, ya, t)
I_T = lambda x, t:  Iy(x, yb, t)

lambda_si = 0.0  

# ============================================================
# Thomas solver (tridiagonal)
# ============================================================

def LU_Thomas(a, b, c, r):
    n = len(r)
    w = zeros(n, float)
    l = zeros(n, float)
    u = zeros(n, float)
    z = zeros(n, float)

    u[0] = b[0]
    for k in range(1, n):
        l[k] = a[k] / u[k-1]
        u[k] = b[k] - l[k] * c[k-1]

    z[0] = r[0]
    for k in range(1, n):
        z[k] = r[k] - l[k] * z[k-1]

    w[n-1] = z[n-1] / u[n-1]
    for k in range(n-2, -1, -1):
        w[k] = (z[k] - c[k] * w[k+1]) / u[k]

    return w

# ============================================================
# Ghost-point Neumann Boundary
# ============================================================
def neumann_x(a, c, r, h, l, L, R):
    # left boundary
    c[0] *= 2.0
    r[0] += 2.0 * l*h*L

    # right boundary
    a[-1] *= 2.0
    r[-1] += 2.0 * l*h*R


def neumann_y(a, c, r, h, l, B, T):
    # bottom boundary
    c[0] *= 2.0
    r[0] += 2.0 * l*h*B

    # top boundary
    a[-1] *= 2.0
    r[-1] += 2.0 * l*h*T


def solve(h, dt):

    iter = int(round(1/dt))
    dt   = 1.0/iter
    l = dt/(2*h**2)

    nx = int((xb-xa)/h) - 1
    ny = int((yb-ya)/h) - 1

    x = np.linspace(xa, xb, nx+2)
    y = np.linspace(ya, yb, ny+2)

    s0 = zeros((nx+2, ny+2), float)
    i0 = zeros((nx+2, ny+2), float)
    s1 = zeros((nx+2, ny+2), float)
    i1 = zeros((nx+2, ny+2), float)

    # Initial Calculation
    for i in range(nx+2):
        for j in range(ny+2):
            s0[i, j] = s_init(x[i], y[j])
            i0[i, j] = i_init(x[i], y[j], dt)

    # Exact Calculation
    s_e = zeros((nx+2, ny+2), float)
    i_e = zeros((nx+2, ny+2), float)
    for i in range(nx+2):
        for j in range(ny+2):
            s_e[i, j] = s_exact(x[i], y[j], 1.0)
            i_e[i, j] = i_exact(x[i], y[j], 1.0)

    # ============================================================
    # Implement Peaceman-Rachford ADI scheme
    # ============================================================

    t = 0.0
    for k in range(iter):
        
        # ------------- calculation for I ---------------

        # x-sweep (I)
        for j in range(ny+2):
            a = zeros(nx+2); b = zeros(nx+2); c = zeros(nx+2); r = zeros(nx+2)

            for i in range(nx+2):
                a[i] = -l if i > 0    else 0.0
                c[i] = -l if i < nx+1 else 0.0

                b[i] = 1.0 + (dt/4.0)*(lambda_si - s0[i, j]) + 2.0*l

                if j > 0: ym = i0[i, j-1]
                else: ym = i0[i, 1] + 2.0 * h * I_B(x[i], t-0.5*dt) # bottom ghost

                if j < ny+1: yp = i0[i, j+1]
                else: yp = i0[i, ny] + 2.0 * h * I_T(x[i], t-0.5*dt) # top ghost

                r[i] = (1.0 - (dt/4.0)*(lambda_si - s0[i, j]) - 2.0*l) * i0[i, j] \
                       + l*(ym + yp) + 0.5*dt * g(x[i], y[j], t)

            neumann_x(a, c, r, h, l, L=I_L(y[j], t), R=I_R(y[j], t))
            i1[:, j] = LU_Thomas(a, b, c, r)

        # y-sweep (I)
        for i in range(nx+2):
            a = zeros(ny+2); b = zeros(ny+2); c = zeros(ny+2); r = zeros(ny+2)

            for j in range(ny+2):
                a[j] = -l if j > 0    else 0.0
                c[j] = -l if j < ny+1 else 0.0

                b[j] = 1.0 + (dt/4.0)*(lambda_si - s0[i, j]) + 2.0*l

                if i>0: xm = i1[i-1, j]
                else: xm = i1[1, j] + 2.0 * h * I_L(y[j], t) # left ghost

                if i<nx+1: xp = i1[i+1,j]
                else: xp = i1[nx, j] + 2.0 * h * I_R(y[j], t) # right ghost

                r[j] = (1.0 - (dt/4.0)*(lambda_si - s0[i, j]) - 2.0*l) * i1[i, j] \
                       + l*(xm + xp) + 0.5*dt * g(x[i], y[j], t)
            

            neumann_y(a, c, r, h, l, B=I_B(x[i], t+0.5*dt), T=I_T(x[i], t+0.5*dt))
            i0[i, :] = LU_Thomas(a, b, c, r)

        # ------------- calculation for S --------------- 

        # x-sweep
        for j in range(ny+2):
            a = zeros(nx+2); b = zeros(nx+2); c = zeros(nx+2); r = zeros(nx+2)

            for i in range(nx+2):
                a[i] = -l if i > 0    else 0.0
                c[i] = -l if i < nx+1 else 0.0

                b[i] = 1.0 + (dt/4.0)*i0[i, j] + 2.0*l


                if j > 0:  ym = s0[i, j-1]
                else: ym = s0[i, 1] + 2.0 * h * S_B(x[i], t) # bottom ghost

                if j < ny+1: yp = s0[i, j+1]
                else: yp = s0[i, ny] + 2.0 * h * S_T(x[i], t) # top ghost

                r[i] = (1.0 - (dt/4.0)*i0[i, j] - 2.0*l) * s0[i, j] \
                       + l*(ym + yp) + 0.5*dt * f(x[i], y[j], t+0.5*dt)

            neumann_x(a, c, r, h, l, L=S_L(y[j], t+0.5*dt), R=S_R(y[j], t+0.5*dt))
            
            s1[:, j] = LU_Thomas(a, b, c, r)

        # y-sweep
        for i in range(nx+2):
            a = zeros(ny+2); b = zeros(ny+2); c = zeros(ny+2); r = zeros(ny+2)

            for j in range(ny+2):
                a[j] = -l if j > 0    else 0.0
                c[j] = -l if j < ny+1 else 0.0

                b[j] = 1.0 + (dt/4.0)*i0[i, j] + 2.0*l

                if i>0: xm = s1[i-1, j]
                else: xm = s1[1, j] + 2.0 * h * S_L(y[j], t+0.5*dt) # left ghost

                if i<nx+1: xp = s1[i+1,j]
                else: xp = s1[nx, j] + 2.0 * h * S_R(y[j], t+0.5*dt) # right ghost

                r[j] = (1.0 - (dt/4.0)*i0[i, j] - 2.0*l) * s1[i, j] \
                       + l*(xm + xp) + 0.5*dt * f(x[i], y[j], t+0.5*dt)

            neumann_y(a, c, r, h, l, B=S_B(x[i], t+dt), T=S_T(x[i], t+dt))
            s0[i, :] = LU_Thomas(a, b, c, r)

        t +=dt
        print('\n t:', t)
        
    # # ------------- 1/2 step calculation for I ---------------
    # x-sweep (I)
    ih = zeros((nx+2, ny+2), float)
    for j in range(ny+2):
        a = zeros(nx+2); b = zeros(nx+2); c = zeros(nx+2); r = zeros(nx+2)

        for i in range(nx+2):
            a[i] = -l if i > 0    else 0.0
            c[i] = -l if i < nx+1 else 0.0

            b[i] = 1.0 + (dt/4.0)*(lambda_si - s0[i, j]) + 2.0*l

            if j > 0: ym = i0[i, j-1]
            else: ym = i0[i, 1] + 2.0 * h * I_B(x[i], t-0.5*dt) # bottom ghost

            if j < ny+1: yp = i0[i, j+1]
            else: yp = i0[i, ny] + 2.0 * h * I_T(x[i], t-0.5*dt) # top ghost


            r[i] = (1.0 - (dt/4.0)*(lambda_si - s0[i, j]) - 2.0*l) * i0[i, j] \
                   + l*(ym + yp) + 0.5*dt * g(x[i], y[j], t)

        neumann_x(a, c, r, h, l, L=I_L(y[j], t), R=I_R(y[j], t))
        ih[:, j] = LU_Thomas(a, b, c, r)

    # y-sweep (I)
    for i in range(nx+2):
        a = zeros(ny+2); b = zeros(ny+2); c = zeros(ny+2); r = zeros(ny+2)

        for j in range(ny+2):
            a[j] = -l if j > 0    else 0.0
            c[j] = -l if j < ny+1 else 0.0

            b[j] = 1.0 + (dt/4.0)*(lambda_si - s0[i, j]) + 2.0*l

            if i>0: xm = ih[i-1, j]
            else: xm = ih[1, j] + 2.0 * h * I_L(y[j], t) # left ghost

            if i<nx+1: xp = ih[i+1,j]
            else: xp = ih[nx, j] + 2.0 * h * I_R(y[j], t) # right ghost


            r[j] = (1.0 - (dt/4.0)*(lambda_si - s0[i, j]) - 2.0*l) * ih[i, j] \
                   + l*(xm + xp) + 0.5*dt * g(x[i], y[j], t)
        

        neumann_y(a, c, r, h, l, B=I_B(x[i], t+0.5*dt), T=I_T(x[i], t+0.5*dt))
        i1[i, :] = LU_Thomas(a, b, c, r)


    # RMS Error
    rms_s = np.sqrt(np.mean((s0 - s_e)**2))
    rms_i = np.sqrt(np.mean(((i0+i1)*0.5 - i_e)**2))

    return rms_s, rms_i


# ============================================================
# Convergence tests
# ============================================================
if __name__ == "__main__":

    save_file = open('ghost_BC.csv', 'w')

    print("\nNeumann BC spatial convergence test (fixed dt)")
    print("------------------------------------------------")

    dt = 1e-4
    h_arr = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
    rmsS2, rmsI2, times2 = [], [], []

    for hh in h_arr:
        start = timer()
        rS, rI = solve(hh, dt)
        end = timer()

        rmsS2.append(rS)
        rmsI2.append(rI)
        times2.append(end - start)

        print(
            f"h={hh:7.5f}, dt={dt:.6f}, "
            f"RMS(S)={rS:.8e}, RMS(I)={rI:.8e}, time={end-start:.2f}s"
        )

    save_file.write("\nSpatial convergence (fixed dt)\n")
    save_file.write("h,dt,RMS of S,RMS of I,order of S,order of I,seconds\n")

    # first row (no order)
    save_file.write(
        f"{h_arr[0]},{dt},{rmsS2[0]},{rmsI2[0]},NaN,NaN,{times2[0]}\n"
    )

    print("\nEstimated spatial orders:")
    for k in range(len(h_arr) - 1):
        oS = np.log2(rmsS2[k] / rmsS2[k + 1])
        oI = np.log2(rmsI2[k] / rmsI2[k + 1])

        print(
            f"h {h_arr[k]} -> {h_arr[k+1]} : "
            f"order(S)≈{oS:.2f}, order(I)≈{oI:.2f}"
        )

        save_file.write(
            f"{h_arr[k+1]},{dt},{rmsS2[k+1]},{rmsI2[k+1]},"
            f"{oS},{oI},{times2[k+1]}\n"
        )

    
    print("Neumann BC temporal convergence test (fixed h)")
    print("------------------------------------------------")
    
    h = 1/256
    dt_arr = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64]

    rmsS, rmsI, times = [], [], []

    for dt in dt_arr:
        start = timer()
        rS, rI = solve(h, dt)
        end = timer()

        rmsS.append(rS)
        rmsI.append(rI)
        times.append(end - start)

        print(
            f"h={h:7.5f}, dt={dt:.6f}, "
            f"RMS(S)={rS:.8e}, RMS(I)={rI:.8e}, time={end-start:.2f}s"
        )

    save_file.write("Temporal convergence (fixed h)\n")
    save_file.write("h,dt,RMS of S,RMS of I,order of S,order of I,seconds\n")

    # first row (no order)
    save_file.write(
        f"{h},{dt_arr[0]},{rmsS[0]},{rmsI[0]},NaN,NaN,{times[0]}\n"
    )

    print("\nEstimated temporal orders:")
    for k in range(len(dt_arr) - 1):
        oS = np.log2(rmsS[k] / rmsS[k + 1])
        oI = np.log2(rmsI[k] / rmsI[k + 1])

        print(
            f"dt {dt_arr[k]} -> {dt_arr[k+1]} : "
            f"order(S)≈{oS:.2f}, order(I)≈{oI:.2f}"
        )

        save_file.write(
            f"{h},{dt_arr[k+1]},{rmsS[k+1]},{rmsI[k+1]},"
            f"{oS},{oI},{times[k+1]}\n"
        )

    save_file.close()

