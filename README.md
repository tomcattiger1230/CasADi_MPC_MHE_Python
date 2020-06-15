# CasADi_MPC_MHE_Python

This repository is an implementation of the work from Mohamed W. Mehrez. I convert the original code from MATLAB to the Python. His videos can be found in Youtube [list](https://www.youtube.com/playlist?list=PLK8squHT_Uzej3UCUHjtOtm5X7pMFSgAL), and his codes in MATLAB are given in his [github](https://github.com/MMehrez/MPC-and-MHE-implementation-in-MATLAB-using-Casadi).

## Environments

- python 3.8 (it should work ab 3.5 or 2.7)
- CasADi == 3.5.1

## Some notations

1. The file name ended with 'opt' using the API from casadi.Opti() to solve the problem.
2. The file name ended with 'struct' using the casadi.tools API to form the problem.
3. The file name without special indication indicates the default SX is utilized.
4. Basically one can also use MX to form the problem. However, with my tests, SX has better performance (in speed)  than MX.