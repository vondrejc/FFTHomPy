Numerical experiments 
========
Folder [examples/sparse](#examples/sparse) contains files working with low-rank tensor implementation of scalar homogenisation problem described in paper:

 - J. Vondřejc, Liu, D, Ladecký, M., and Matthies, H. G.: *FFT-Based Homogenisation Accelerated by Low-Rank Approximations.* 2019. arXiv:1902.07455

## Model problem
In file [diffusion.py](#diffusion.py) a model scalar elliptic homogenisation problem described in section 2.1. of the paper is implemented.
For predefined:

 -  dimension (dim= 2 or 3),
 - grid size (N= odd number),
 - material (material = 0 - square inclusion,
    1 - pyramid inclusion, 2 - stochastic material),
 - low-rank tensor format (kind= 0-canonical, 1- Tucker, 2- Tensor-Train)
  
 [diffusion.py](#diffusion.py) compute one element of homogenised material property of the material (see section 2.2. of the paper).
 The problem solution is computed by two different approaches: 
  
   - (Ga) Galerkin approximation ,
   - (GaNi) Galerkin approximation with Numerical Integration,
 
  and two different formats:
  
   - full tensor format,
   - low-rank format (canonical, Tucker or Tensor-Train).

Materials are predefined in [material_setting.py](#material_setting.py).

##Generate results 

File [diffusion_comparison.py](#diffusion_comparison.py) computes problem defined
in [diffusion.py](#diffusion.py) for different parameters (mesh size-N, rank-r, low-rank format, material, ...) and compare solver behavior
  (error evolution, memory consumption, and evolution of residua) with full tensor approach. For more details see section 4.2. and 4.3. of the paper.

File [experiment_time_efficiency.py](#experiment_time_efficiency.py) use problem defined in [diffusion.py](#diffusion.py) with material 0 (squere inclusion). 
This file computes the computational time at the same level of accuracy for the scheme with exact integration (Ga). The full solution is calculated on a grid of size (N,...,N) while the low-rank solution on the grid (3N,...,3N) with a solution rank to achieve the same level of accuracy as full scheme. For more details see section 4.4. of the paper.


##Plot results

File [plots.py](#plots.py) contains procedures which creates .pdf figures with results.
Procedures plot_error(), plot_memory() and plot_residuals() use data genereted by [diffusion_comparison.py](#diffusion_comparison.py) 
 and plot_time() uses data genereted by [experiment_time_efficiency.py](#experiment_time_efficiency.py).
Visual style, lines and labels are defined in [fig_pars.py](#fig_pars.py).


