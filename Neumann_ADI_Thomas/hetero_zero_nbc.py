import numpy as np
from numpy import zeros
from math import exp, sin, cos, pi
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Exact solutions
# ============================================================
s_exact = lambda x,y,t: cos(pi*x) * cos(pi*y) * exp(t)
i_exact = lambda x,y,t: cos(2*pi*x) * cos(pi*y) * cos(t)

# Heterogeneous coefficients
kS = lambda x,y: 1.0 + 0.5*sin(2*pi*x)*sin(2*pi*y)
kI = lambda x,y: 1.0 + 0.3*cos(2*pi*x)
lambda_si = 1.0

# Initial conditions
s_init = lambda x,y: cos(pi*x) * cos(pi*y)                      # S at t=0
i_init = lambda x,y,dt: cos(2*pi*x) * cos(pi*y) * cos(-0.5*dt)  # I at t=-dt/2

# Source terms
f = lambda x, y, t: (2 * pi**2 + 1 + i_exact(x,y,t)) * s_exact(x,y,t)
g = lambda x, y, t: (5 * pi**2 - s_exact(x,y,t)) * i_exact(x,y,t) - cos(2*pi*x) * cos(pi*y) * sin(t)

# ============================================================
# Manufactured source terms
# ============================================================
def f(x, y, t):
    S = s_exact(x, y, t)
    I = i_exact(x, y, t)
    return S + 2*pi*pi*S + kS(x, y)*I*S

def g(x, y, t):
    S = s_exact(x, y, t)
    I = i_exact(x, y, t)
    It = -sin(t)*cos(2*pi*x)*cos(pi*y)
    return It + 5*pi*pi*I + kI(x, y)*(lambda_si - S)*I

# Domain
xa, xb = 0.0, 1.0
ya, yb = 0.0, 1.0


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
# Neumann BC (ghost-point reflection for zero-flux)
# ============================================================
def neumann_x(a, c):
    c[0] *= 2.0
    a[-1] *= 2.0

def neumann_y(a, c):
    c[0] *= 2.0
    a[-1] *= 2.0

def solve(h, dt):

    iter = int(round(1/dt))
    dt   = 1.0/iter
    l = dt/(2*h*h)

    nx = int((xb - xa)/h) - 1
    ny = int((yb - ya)/h) - 1

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

                coeff = kI(x[i], y[j]) * (lambda_si - s0[i, j])
                b[i] = 1.0 + (dt/4.0)*coeff + 2.0*l

                ym = i0[i, j-1] if j > 0 else i0[i, 1]      # bottom ghost
                yp = i0[i, j+1] if j < ny+1 else i0[i, ny]  # top ghost

                r[i] = (1.0 - (dt/4.0)*coeff - 2.0*l) * i0[i, j] \
                       + l*(ym + yp) + 0.5*dt * g(x[i], y[j], t)

            neumann_x(a, c)
            i1[:, j] = LU_Thomas(a, b, c, r)

        # y-sweep (I)
        for i in range(nx+2):
            a = zeros(ny+2); b = zeros(ny+2); c = zeros(ny+2); r = zeros(ny+2)

            for j in range(ny+2):
                a[j] = -l if j > 0    else 0.0
                c[j] = -l if j < ny+1 else 0.0

                coeff = kI(x[i], y[j]) * (lambda_si - s0[i, j])
                b[j] = 1.0 + (dt/4.0)*coeff + 2.0*l

                xm = i1[i-1, j] if i > 0 else i1[1, j]    # left ghost
                xp = i1[i+1,j] if i < nx+1 else i1[nx, j] # right ghost

                r[j] = (1.0 - (dt/4.0)*coeff - 2.0*l) * i1[i, j] \
                       + l*(xm + xp) + 0.5*dt * g(x[i], y[j], t)
            
            neumann_y(a, c)
            i0[i, :] = LU_Thomas(a, b, c, r)

        # ------------- calculation for S --------------- 
        # x-sweep
        for j in range(ny+2):
            a = zeros(nx+2); b = zeros(nx+2); c = zeros(nx+2); r = zeros(nx+2)

            for i in range(nx+2):
                a[i] = -l if i > 0    else 0.0
                c[i] = -l if i < nx+1 else 0.0

                coeff = kS(x[i], y[j])*i0[i, j]
                b[i] = 1.0 + (dt/4.0)*coeff + 2.0*l

                ym = s0[i, j-1] if j > 0 else s0[i, 1]     # bottom ghost
                yp = s0[i, j+1] if j < ny+1 else s0[i, ny] # top ghost

                r[i] = (1.0 - (dt/4.0)*coeff - 2.0*l) * s0[i, j] \
                       + l*(ym + yp) + 0.5*dt * f(x[i], y[j], t+0.5*dt)

            neumann_x(a, c)
            s1[:, j] = LU_Thomas(a, b, c, r)

        # y-sweep
        for i in range(nx+2):
            a = zeros(ny+2); b = zeros(ny+2); c = zeros(ny+2); r = zeros(ny+2)

            for j in range(ny+2):
                a[j] = -l if j > 0    else 0.0
                c[j] = -l if j < ny+1 else 0.0

                coeff = kS(x[i], y[j])*i0[i, j]
                b[j] = 1.0 + (dt/4.0)*coeff + 2.0*l

                xm = s1[i-1, j] if i > 0 else s1[1, j]  # left ghost
                xp = s1[i+1,j] if i<nx+1 else s1[nx, j] # right ghost

                r[j] = (1.0 - (dt/4.0)*coeff - 2.0*l) * s1[i, j] \
                       + l*(xm + xp) + 0.5*dt * f(x[i], y[j], t+0.5*dt)

            neumann_y(a, c)
            s0[i, :] = LU_Thomas(a, b, c, r)

        t +=dt
        print('\nt:',t)

  
    # # ------------- 1/2 step calculation for I ---------------
    # x-sweep (I)
    ih = zeros((nx+2, ny+2), float)
    for j in range(ny+2):
        a = zeros(nx+2); b = zeros(nx+2); c = zeros(nx+2); r = zeros(nx+2)

        for i in range(nx+2):

            a[i] = -l if i > 0    else 0.0
            c[i] = -l if i < nx+1 else 0.0

            coeff = kI(x[i], y[j]) * (lambda_si - s0[i, j])
            b[i] = 1.0 + (dt/4.0)*coeff + 2.0*l

            ym = i0[i, j-1] if j > 0 else i0[i, 1]      # bottom ghost
            yp = i0[i, j+1] if j < ny+1 else i0[i, ny]  # top ghost

            r[i] = (1.0 - (dt/4.0)*coeff - 2.0*l) * i0[i, j] \
                   + l*(ym + yp) + 0.5*dt * g(x[i], y[j], t)

        neumann_x(a, c)
        ih[:, j] = LU_Thomas(a, b, c, r)

    # y-sweep (I)
    for i in range(nx+2):
        a = zeros(ny+2); b = zeros(ny+2); c = zeros(ny+2); r = zeros(ny+2)

        for j in range(ny+2):

            a[j] = -l if j > 0    else 0.0
            c[j] = -l if j < ny+1 else 0.0

            coeff = kI(x[i], y[j]) * (lambda_si - s0[i, j])
            b[j] = 1.0 + (dt/4.0)*coeff + 2.0*l

            xm = ih[i-1, j] if i > 0 else ih[1, j]    # left ghost
            xp = ih[i+1,j] if i < nx+1 else ih[nx, j] # right ghost

            r[j] = (1.0 - (dt/4.0)*coeff - 2.0*l) * ih[i, j] \
                   + l*(xm + xp) + 0.5*dt * g(x[i], y[j], t)
    

        neumann_y(a, c)
        i1[i, :] = LU_Thomas(a, b, c, r)


    # RMS Error
    rms_s = np.sqrt(np.mean((s0 - s_e)**2))
    rms_i = np.sqrt(np.mean(((i0+i1)*0.5 - i_e)**2))

    return rms_s, rms_i

# ============================================================
# Convergence tests
# ============================================================
if __name__ == "__main__":

    save_file = open('heterogenity.csv', 'w')

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

