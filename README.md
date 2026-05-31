# Infectious Disease Solver

Numerical solvers for infectious disease models based on systems of partial differential equations (PDEs).

## Overview

This repository contains finite difference implementations for solving coupled infectious disease PDE systems. The project focuses on developing and analyzing stable and accurate numerical methods for epidemiological models.

Implemented methods include:

* Alternating Direction Implicit (ADI) Scheme
* Peaceman–Rachford ADI Method
* Implicit Finite Difference Method
* Gauss–Seidel Iteration
* Successive Over-Relaxation (SOR)
* Convergence and Error Analysis

## Repository Structure

```text
infectious-disease-solver/
│
├── ADI Scheme/
│   ├── ADI.py
│   └── ADI_result.csv
│
├── CN Staggered Scheme/
│   ├── SOR_Method.py
│   └── SOR_result.csv
│
├── Implicit Scheme/
│   ├── Implicit scheme (Gauss-Seidel).py
│   └── Implicit scheme (SOR).py
│
└── README.md
```

## Mathematical Model

The code solves coupled reaction-diffusion equations representing infectious disease dynamics:

[
\frac{\partial S}{\partial t}
=============================

\nabla^2 S + f(S,I),
]

[
\frac{\partial I}{\partial t}
=============================

\nabla^2 I + g(S,I),
]

where:

* (S(x,y,t)) denotes the susceptible population,
* (I(x,y,t)) denotes the infected population,
* diffusion terms model spatial movement,
* reaction terms describe disease transmission and recovery.

Manufactured exact solutions are used to verify numerical accuracy and convergence.

## Features

* Two-dimensional spatial domain simulations
* Dirichlet boundary conditions
* Error computation against exact solutions
* Spatial convergence studies
* Temporal convergence studies
* Runtime performance measurements

## Requirements

```bash
pip install numpy scipy matplotlib pandas
```

## Running the Codes

### ADI Solver

```bash
python ADI.py
```

### Implicit SOR Solver

```bash
python "Implicit scheme (SOR).py"
```

### Implicit Gauss-Seidel Solver

```bash
python "Implicit scheme (Gauss-Seidel).py"
```

## Applications

* Infectious disease spread modeling
* Epidemiological PDE simulations
* Numerical analysis research
* Finite difference method validation
* Graduate-level computational mathematics projects

## Author

**Md Abu Talha**

Ph.D. Candidate, Department of Mathematics
Southern Methodist University (SMU)
Dallas, Texas, USA

## License

MIT License
