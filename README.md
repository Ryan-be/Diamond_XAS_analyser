# Diamond_XAS_analyser
This Python (3.7) program is designed to be used to analyse the pre-edge peak in XAS data from the Diamond Light Source (UK's national synchrotron). It is designed to work for iron energies, however, it would be very simple to just change the energy at which the data sets are cut to work with other energies.
It will atomically indexes all of the spectra files in a folder. It then analyses then normalises then plots the data.

Most of the program is concerned with with manipulation of data and model fitting. In order to use the code three steps need to be followed:
  1. Install the lmfit package. This can be done on any computer which has the pip installer which is available for Linux,          Windows and Mac OS. In any Terminal type: pip install lmfit
  2. Modify the "Path" lines of code to point to a file containing all of the data as well as the line: while ind <= N, where      N is the number of data files.
  3. lastly, run the program and all relevant plots will be saved to the current working directory.
