This code enables the users to approximate the probability distribution p(s) of a grid of spin in one and two dimensions. 

It can be used to solve the Ising problem in 2 dimensions. 

The ising1d.py file trains a model to approximate a 1D distribution that can be changed 
and then computes the probability to observe some spins according to the approximed distribution.

The ising2d.py file approximates the probability distribution of the ising problem in two dimensions at a fixed temperature. 
It saves samples of magnetization and can plot the evolution of the magnetization of a given grid of spins with temperature. 
It can plot those graphs with and without annealing. 

This code comes form a project at Ecole Polytechnique that aims to reproduce the results from ·	Reproducing the results of Wu, Dian, Lei Wang, and Pan Zhang. "Solving statistical mechanics using variational autoregressive networks." Physical review letters 122.8

