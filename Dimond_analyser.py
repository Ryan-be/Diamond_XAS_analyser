
# coding: utf-8

# brfore running this code lmfit must be installed: to so this go to the console and exicute the following line of code -> 'pip install lmfit'

# # Imports

# In[1]:


import math
import sys
import glob
import errno
import matplotlib.pyplot as plt
import os 
from scipy.interpolate import interp1d
import numpy as np
import csv
import pandas as pd
from scipy.optimize import curve_fit
import plotly.plotly as py
import plotly.graph_objs as go
import glob 
import pylab as plb
from scipy import asarray as ar,exp
from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model


# ## Index all of the files in the folder and save it as a .txt file "index_files.txt"

# In[2]:



path = '/Users/ryan_be/test/*'   #Define the parth to the containing folder that 
                               #contains the data files. Star needs to be included here.
path2 = '/Users/ryan_be/test/'   #Define the parth to the containing folder that 
                               #contains the data files. No star needed.                            
path3 = '/Users/ryan_be/plots/'
files=glob.glob(path)   
#f = open('index_files.txt', 'w')  #Create an empty text file
#f.close()

#The following loop creates an index of all of the file paths in the data containing 
#directory. It then writes them to the .txt file "index_files.txt"
for file in files: 
  f=open("index_files.txt", "a+")
  f.write(file)
  f.write('\n')
  f.close()

  #print (file)
with open('index_files.txt', 'r') as program:
  data = program.readlines()

####################################################
# The following short code block, when uncommented,#
# will add an index columb to the file index .txt  #
# file.                                            #
####################################################

#with open('index_files.txt', 'w') as program:
#    for (number, line) in enumerate(data):
#        program.write('%d  %s' % (number + 1, line))

####################################################
# The following function will remove the file      #
# extention for use in naming the plots later in   #
# the program                                      #
####################################################

def file_base_name(file_name):
  if '.' in file_name:
      separator_index = file_name.index('.')
      separator_index2 = file_name.index('11')
      base_name = file_name[:separator_index]
      base_name = base_name[separator_index2:]
      return base_name
  else:
      return file_name
  
#with open('index_files.txt', 'r') as program:
# data = program.readlines()

#with open('index_files.txt') as fp:  
#   line = fp.readline()
#   cnt = 1
#   while line:
#       print("Line {}: {}".format(cnt, line.strip()))
#       line = fp.readline()
#       cnt += 1
f.close()




f = open('index_files.txt', 'r')  # We need to re-open the file
line = f.readlines()
print(line) 
f.close()


# ## Smooth the data
ind = 0 
while ind <= 46:

    file = files[ind]
    f1 = open(file, 'r')
    
    d = np.loadtxt(f1)
    x = d[:,0]    #Define x to be columb 1, this is the Energy 
    y = d[:,7]    #Define y to be columb 8, this is FF/I0 i.e the normalised data from athena
    #y = d[:,1]   use with .nor files#
    sample_name = file_base_name(file)  #This is the name of the sample taken from the file name
    plt.ylabel('Normalised Intensity')
    plt.xlabel('Energy /ev')
    plt.title(sample_name)
    plt.show()
    plt.plot(x,y)
   
    plt.savefig(path3+sample_name+'.png', dpi=900)
    plt.close('all')
    print(sample_name)

####################################################
# The following code block smooths the data to     #
# eliminate any variations in the bulk trends      #
####################################################

#f = interp1d(x, y)
    f = interp1d(x, y, bounds_error=False, kind='cubic')


    f2 = interp1d(x, y, kind='cubic', bounds_error=False)
    xnew = np.arange(7000, 7500, 0.1)     #np.linspace(7000, 7500, num=1000000)
#plt.plot(x,y, xnew, f(xnew), '-')
    plt.plot(x, y,'.', xnew, f(xnew), '-', xnew, f2(xnew), '--')

#plt.legend(['data', 'linear', 'cubic'], loc='best')

    plt.savefig(sample_name+'full_data_with_smooth_function.png', dpi=900)
    plt.show()
    plt.close('all')
    

# ## Save the smoothed data as a CSV




    data = zip(xnew,f2(xnew))
    smoothname = str('Smoothed'+str(ind)+'.csv')
    

    with open(smoothname, 'w') as f: 
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(xnew,f2(xnew)))





# ## Cut of pre edge
# #### This is done rather arbaterally at y = 0.5 (this is based on Leons orriginal code)
# 



    plt.close('all') #This ensures there is no overplotting 

####################################################
# The following code block loads the data in the   #
# CSV file into a pandas data frame.               # 
####################################################

    smoothdat = pd.DataFrame(  
            {'E': xnew,
             'Flat': f2(xnew)
             })  

# count_nonzero removes all rows with a NAN value in the columb
    x1 = np.count_nonzero(smoothdat < 6)#cut off the pre-edge at y = 0.5 
    x2 = np.count_nonzero(smoothdat)

    predge = smoothdat[0:x1]
    plt.plot(predge['E'], predge['Flat'])
    plt.show()
    plt.close('all')
    

# ### Remove the predge feture data



    x1 = np.count_nonzero(predge['E'] < 7105)
    x2 = np.count_nonzero(predge['E'] > 7115)
    x3 = np.count_nonzero(predge['E'] < 7120)
    xref = np.count_nonzero(predge['E'])
    x2 = xref - x2
    prepredge = predge[1000:x1]
    predgepost = predge[x2:x3] 

    frames = [prepredge, predgepost]
    xfeature = predge[x1:x2]


    newa = pd.concat(frames)
    plt.plot(newa['E'], newa['Flat'], '.')
    plt.show()
    plt.close('all')
    

# ### Fit polynomial and expanential
# ##### This was the bases for the orriginal moddel which was based soley on inspection, explanential and quadratic. However it quickly became evident that the models where not suitable as the did not converge 



#These moddles where tried again using diffrent syntax but to no avail 
    x = newa['E']
    y = newa['Flat']
    x.dropna(inplace=True)
    y.dropna(inplace=True)
    x= x
    y=y

# #### A simple linear model was tried simply for code and logic testing, this gave supprisingly good results. However the model does not match the physics.



    xData = xfeature['E']
    yData = xfeature['Flat']
    xData.dropna(inplace=True)
    yData.dropna(inplace=True)

    print(xData.min(), xData.max())
    print(yData.min(), yData.max())

#    fitted = np.polyfit(xData, yData, 1)[::-1]
#    y = np.zeros(len(xData))
#    for i in range(len(fitted)):
#        y += fitted[i]*xData**i
#        plt.plot(xfeature['E'],xfeature['Flat'],'.',xData, y )
#        x = xData
#        y =yData-y
#        plt.plot(x, y,'.')
#        #plt.show()
#        plt.close('all')
    


    x = predge['E']*0.001#xData
    y = predge['Flat']#yData
    x = x[1:,]
    y = y[1:,]
    x.dropna(inplace=True)
    y.dropna(inplace=True)
    xmod = np.linspace(7.1, 7.12, 96)


#data = loadtxt('model1d_gauss.dat')


    def gaussian(x, amp, cen, wid, const):
        "1-d gaussian: gaussian(x, amp, cen, wid)"
        return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2) +const )

    gmodel = Model(gaussian)
    result = gmodel.fit(y, x=x, amp=5, cen=5, wid=1, const = 0.1)

    print(result.fit_report())

    plt.plot(x, y,         'bo')
    plt.plot(x, result.init_fit, 'k--')
    plt.plot(x, result.best_fit, 'r-')
    plt.show()

    plt.close('all')
    

    ynew = y- result.best_fit
    xmodcomp = result.best_fit[x1:x2]
    ynew = ynew[x1:x2]
    y = y[x1:x2]
    x = x[x1:x2]*1000
    plt.plot(x, ynew, label = 'Flattened data')
    plt.plot(x, y, label = 'Data')
    plt.plot(x, xmodcomp,label = 'Pre-edge model')
    plt.title(sample_name+'normilised')
    maxy = max(ynew)
    maxx= x[ynew.argmax()]
    plt.axvline(x=maxx)
    plt.legend()
    name = (sample_name+'maxatnormilised_with_lines_txt.png')
    plt.text(maxx, maxy, 'Max y occures at x=' and maxx)
    plt.ylabel('Normilised Intensity', fontsize=11)
    
    plt.xlabel('Energy /eV', fontsize=11)
    plt.savefig(name, dpi=900)
    plt.show()
    plt.close('all')
    ind = ind + 1
