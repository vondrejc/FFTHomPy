FFTHomPy
========

FFT-based homogenization in Python is a numerical software for evaluating guaranteed upper-lower bounds on homogenized properties. The algorithms implemented here are based on the papers in [references](#references) .

## News

- The code now contains modelling using tesors with a low rank approximation.

## Manual

The basic manual can be found at
- http://FFTHomPy.bitbucket.io

or downloaded at
- http://FFTHomPy.bitbucket.io/FFTHomPy.pdf

Tutorials can be found in a folder '/tutorial'.

## Requirements and installation

No special installation is required. However, the folder with the code has to be in the python path.

The code is optimised for [Python](https://www.python.org) (version 3.6) and
depends on the following numerical libraries:
- [NumPy](http://www.numpy.org) (version 1.16) and
- [SciPy](https://www.scipy.org) (version 1.3) for scientific computing as well as on the
- [Matplotlib](https://matplotlib.org/) (version 3.1) for plotting
- [StoPy](https://github.com/vondrejc/StoPy) for uncertainty quantification
- [ttpy](https://github.com/oseledets/ttpy) Python implementation of the Tensor Train (TT)-Toolbox

## References

The code is based on the following papers, where you can find more theoretical information.

- J. Vondřejc, Liu, D, Ladecký, M., and Matthies, H. G.: *FFT-Based Homogenisation Accelerated by Low-Rank Approximations.* 2019. arXiv:1902.07455
- J. Zeman, T. W. J. de Geus, J. Vondřejc, R. H. J. Peerlings, and M. G. D. Geers: *A finite element perspective on non-linear FFT-based micromechanical simulations.* International Journal for Numerical Methods in Engineering, 111 (10), pp. 903-926, 2017. arXiv:1601.05970
- N. Mishra, J. Vondřejc, J. Zeman: *A comparative study on low-memory iterative solvers for FFT-based homogenization of periodic media.* Journal of Computational Physics, 321, pp. 151-168, 2016. arXiv:1508.02045
- J. Vondřejc: *Improved guaranteed computable bounds on homogenized properties of periodic media by Fourier-Galerkin method with exact integration.* International Journal for Numerical Methods in Engineering, 107 (13), pp.~1106-1135, 2016. arXiv:1412.2033
- J. Vondřejc, J. Zeman, I. Marek: *Guaranteed upper-lower bounds on homogenized properties by FFT-based Galerkin method.* Computer Methods in Applied Mechanics and Engineering, 297, pp. 258–291, 2015. arXiv:1404.3614
- J. Vondřejc, J. Zeman, I. Marek: *An FFT-based Galerkin method for homogenization of periodic media.* Computers and Mathematics with Applications, 68, pp. 156-173, 2014. arXiv:1311.0089
- J. Zeman, J. Vondřejc, J. Novák and I. Marek: *Accelerating a FFT-based solver for numerical homogenization of periodic media by conjugate gradients.* Journal of Computational Physics, 229 (21), pp. 8065-8071, 2010. arXiv:1004.1122

