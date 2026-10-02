FFTHomPy
========

FFT-based homogenization in Python is a numerical software for evaluating guaranteed upper-lower bounds on homogenized properties. The algorithms implemented here are based on the papers in [references](#references) .

## News

- The code now contains modelling using tesors with a low rank tensor approximation.

## Manual

The basic manual can be found at
- http://FFTHomPy.bitbucket.io

or downloaded at
- http://FFTHomPy.bitbucket.io/FFTHomPy.pdf

Tutorials can be found in a folder '/tutorial'.

## Requirements and installation

The current development version is `1.0.4.dev0`. It supports Python 3.11–3.14
with NumPy 1.26.4 or newer and SciPy 1.11.4 or newer. uv selects releases
compatible with the chosen Python version, including the latest NumPy and
SciPy on Python 3.14. Install the package and run its tests with
[uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run pytest
```

For plotting, use `uv sync --extra plot`. The Tensor Train functionality
requires the legacy [ttpy](https://pypi.org/project/ttpy/) package, which is
not installed by default; its tests are skipped when it is unavailable.
The stochastic material examples additionally require
[StoPy](https://github.com/vondrejc/StoPy). The package and its core tests do
not require either of these legacy dependencies.

## References

The code is based on the following papers, where you can find more theoretical information.

- J. Vondřejc, D. Liu, M. Ladecký, and H.G. Matthies: *FFT-Based Homogenisation Accelerated by Low-Rank Tensor Approximations.* Computer Methods in Applied Mechanics and Engineering, 364, pp. 112890, 2020. https://doi.org/10.1016/j.cma.2020.112890
- J. Vondřejc, T.W.J. de Geus: Energy-based comparison between the Fourier--Galerkin method and the finite element method. Journal of Computational and Applied Mathematics. 374, pp. 112585, 2020. https://doi.org/10.1016/j.cam.2019.112585
- J. Zeman, T. W. J. de Geus, J. Vondřejc, R. H. J. Peerlings, and M. G. D. Geers: *A finite element perspective on non-linear FFT-based micromechanical simulations.* International Journal for Numerical Methods in Engineering, 111 (10), pp. 903-926, 2017. arXiv:1601.05970
- N. Mishra, J. Vondřejc, J. Zeman: *A comparative study on low-memory iterative solvers for FFT-based homogenization of periodic media.* Journal of Computational Physics, 321, pp. 151-168, 2016. arXiv:1508.02045
- J. Vondřejc: *Improved guaranteed computable bounds on homogenized properties of periodic media by Fourier-Galerkin method with exact integration.* International Journal for Numerical Methods in Engineering, 107 (13), pp.~1106-1135, 2016. arXiv:1412.2033
- J. Vondřejc, J. Zeman, I. Marek: *Guaranteed upper-lower bounds on homogenized properties by FFT-based Galerkin method.* Computer Methods in Applied Mechanics and Engineering, 297, pp. 258–291, 2015. arXiv:1404.3614
- J. Vondřejc, J. Zeman, I. Marek: *An FFT-based Galerkin method for homogenization of periodic media.* Computers and Mathematics with Applications, 68, pp. 156-173, 2014. arXiv:1311.0089
- J. Zeman, J. Vondřejc, J. Novák and I. Marek: *Accelerating a FFT-based solver for numerical homogenization of periodic media by conjugate gradients.* Journal of Computational Physics, 229 (21), pp. 8065-8071, 2010. arXiv:1004.1122
