# Infectious Disease Solver

Numerical solvers for infectious disease models using finite difference methods and partial differential equations (PDEs).

## Repository Structure

```text
infectious-disease-solver/
│
├── CN Staggered Scheme/
│   ├── ADI_Method.py
│   ├── Gauss_Seidel.py
│   ├── SOR_Method.py
│   ├── Matrix_Solver.py
│   ├── ADI_result.csv
│   ├── Gauss_result.csv
│   ├── SOR_result.csv
│   └── Solver_result.csv
│
├── Implicit Scheme/
│   ├── Implicit scheme (Gauss-Seidel).py
│   └── Implicit scheme (SOR).py
│
├── Neumann_ADI_Thomas/
│   ├── ghost_neumann_bc.py
│   ├── ghost_BC.csv
│   ├── hetero_zero_nbc.py
│   ├── heterogenity.csv
│   ├── real1d_cnty_Covid.ipynb
│   └── real2d_TX_Covid.ipynb
│
└── README.md
```

## Overview

This repository contains Python implementations of numerical schemes for infectious disease modeling. The project focuses on solving coupled PDE-based 2D Fisher's models using implicit and ADI-type finite difference methods.

## Main Components

### CN Staggered Scheme

This folder contains Crank–Nicolson staggered scheme implementations using:

* ADI method
* Gauss–Seidel iteration
* SOR iteration
* Matrix-based solver

It also includes CSV files containing numerical results.

### Implicit Scheme

This folder contains fully implicit finite difference solvers implemented with:

* Gauss–Seidel method
* Successive Over-Relaxation method

### Neumann_ADI_Thomas

This folder contains ADI solvers with Neumann boundary conditions using ghost-point methods and Thomas algorithm. It also includes notebook examples using COVID county-level data.

## Requirements

```bash
pip install numpy scipy matplotlib pandas jupyter
```

## Applications

* Infectious disease spread modeling
* Reaction-diffusion PDE simulation
* Numerical method comparison
* Error and convergence analysis
* COVID-19 spatial data modeling

## Author

**Md Abu Talha**
Ph.D. Candidate
Department of Mathematics
Southern Methodist University
Dallas, Texas, USA
