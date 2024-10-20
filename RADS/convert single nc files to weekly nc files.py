#import nco
#ncea  ["20220101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4"] ["20220102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4"] "week1.nc4"

#import nco
#nces -O -analysed_sst 20220101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4 20220102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4 output.nc4


#----------------------


#import subprocess
#from nco import Nco
#nco = Nco()

# Define the names of the input files
#input_files = [r'C:\Users\Naomi\AppData\Roaming\Python\Python310\site-packages\py_eddy_tracker\data\20220101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4', r'C:\Users\Naomi\AppData\Roaming\Python\Python310\site-packages\py_eddy_tracker\data\20220102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4']

# Define the name of the output file
#output_file = 'average.nc'

# Call the nces function to calculate the average
#subprocess.call(['nces', '-O', '-analysed_sst', 'avg', ' '.join(input_files), output_file])


#import nco
#from nco import Nco
#nco = Nco()
#nco.ncea(input=['20220101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4', '20220102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4'], output='average.nc4').variables['analysed_sst']



#--------------------------



#from cdo import Cdo   # python version
#cdo = Cdo()
#cdo.debug = True

#cdo.infov(input=['20220101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4', '20220102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4'])         #python
#cdo.showlevels(input=['20220101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4', '20220102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4'])


#cdo.ensmean([analysed_sst], input=['20220101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4', '20220102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4'], output='average.nc4')


#--------------------


from cdo import Cdo
import glob

cdo = Cdo()
# use a special binary
cdo.setCdo('/sw/rhel6-x64/cdo/cdo-1.9.5-gcc64/bin/cdo')
cdo.ensmean(input = glob.glob(['20220101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4', '20220102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4']), options = '-r', output='average.nc4')