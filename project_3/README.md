## Repository for project 3: FYS-STK4155

### Eigval.py
This will calculate the highest (or lowest) eigenvalue of the symmetric, 6x6 matrix presented in the
if `__name__` block, also computes all the eigenvalues/eigenvectors using np.linalg for comparison.
Code is easily ran as `> python3 eigval.py`

### Fweuler.py
This computes a solution to the 1D heat  diffusion PDE using method of lines to produce ODEs that are solved using Forward Euler, and in the if `__name__` block there are three if blocks 
which are set if False. Change either of these to if True to run the subsequent code, and compute solutions with various 
parameters and produce the plots presented in the report. Run the code using `> python3 fweuler.py`.
