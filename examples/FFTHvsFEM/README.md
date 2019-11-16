 Energy-based comparison between the Fourier-Galerkin method and the finite element
==================================================================================

This folder [examples/FFTHvsFEM](#examples/FFTHvsFEM) contains the implementation comparing the FFT-homogenisation
with the finite element method (FEM). It is based on the following publication:

 - Vondřejc, J., & de Geus, T. W. J. (2019). Energy-based comparison between the Fourier--Galerkin method and the finite element method. Journal of Computational and Applied Mathematics. https://doi.org/10.1016/j.cam.2019.112585


This file explain the basic usage of the attached code that is written in Python3
(https://www.python.org/ using version 3.6).
Each script can be run using command 'python <name of script>', e.g. 'python FEM.py'.
The code depends on the numerical libraries
- NumPy (http://www.numpy.org/ using version 1.17.2),
- SciPy (https://www.scipy.org/ using version 1.3.1), and
- finite element software FEniCS (https://fenicsproject.org/ using version 2019.1.0).


file: FEM.py
------------
Solves the homogenisation problem using the finite element method.


file: FFTH_GaNi.py
------------------
Solves the homogenisation problem using the Fourier-Galerkin method with numerical integration which is fully equivalent to Moulinec-Suquet scheme.


file: FFTH_Ga.py
----------------
Solves the homogenisation problem using the Fourier-Galerkin method with exact integration. It provides the best approximation using trigonometric polynomials.


file: functions.py
------------------
This file contains auxiliary functions.