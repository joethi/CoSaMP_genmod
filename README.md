This repository contains the implementation of a deep learning-based algorithm developed for my master's thesis, titled "Neural Networks Based Algorithm for Polynomial Chaos Expansions." The core algorithm, known as the "Modified Orthogonal Matching Pursuit (OMP)," is designed to enhance the efficiency of Polynomial Chaos Expansions (PCEs). This method has practical applications in the context of NASA's space missions, providing robust solutions for uncertainty quantification in critical aerospace systems.

**The abstract of Thesis:**


Uncertainty Quantification aims at finding the effect of input or parametric variability in physical systems, such as partial differential equations, on the quantity of interests. Polynomial Chaos Expansion (PCE) is an efficient method for propagating the input uncertainty. When the number solution evaluations are small due to the expensive numerical simulations, the problem becomes solving under-determined system of linear equations. The current sparsity promoting techniques, such as orthogonal matching pursuit (OMP), to solve this problem become computationally infeasible for the high dimensional systems due to the curse of dimensionality. Moreover, with the low number of samples these methods perform poorly in terms of accuracy. In this thesis, a sparsity promoting method, called Modified OMP (MO), that uses neural networks was developed to mitigate both issues. Numerical experiments show for the PCE coefficients with a particular structure that MO outperforms the traditional OMP method.


Run the following main script to execute the MO algorithm: "no_ray/scripts/cosamp_GenMod_nu_4_pcklec_v2.py".

