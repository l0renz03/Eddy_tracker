#import nco
#from nco import Nco
#nco = Nco()
#nco.ncea(input='C:/Users/Naomi/AppData/Roaming/Python/Python310/site-packages/py_eddy_tracker/data/GHRSST/202201*.nc4', output='average.nc4').variables['analysed_sst'][:]


# ------ ALTERNATIVE ------


#from cdo import *
#cdo=Cdo()
#cdo.ensmean(input = "C:\\Users\\Naomi\\AppData\\Roaming\\Python\\Python310\\site-packages\\py_eddy_tracker\\data\\GHRSST\\*.nc4", output = "C:\\Users\\Naomi\\AppData\\Roaming\\Python\\Python310\\site-packages\\py_eddy_tracker\\data\\GHRSST\\output.nc4")


# BOTH ALTERNATIVES GIVE A DIFFERENT ERROR


# ------ ALTERNATIVE ------
import os
import subprocess
bashCommand = "cdo ensmean C:/Users/Naomi/AppData/Roaming/Python/Python310/site-packages/py_eddy_tracker/data/GHRSST/2022*.nc4 C/Users/Naomi/AE2224-I_D01/GHRSST/SST_weekly/output_6.nc4"
#bashCommand = "cdo ensmean GHRSST/SST_data/2022*.nc4 GHRSST/SST_weekly/output_6.nc"
#subprocess.run(bashCommand)
os.system(bashCommand)

# ------ ALTERNATIVE ------
#import nco
#from nco import Nco
#nco = Nco()
#nco.ncea(input='C:\\Users\\Naomi\\AppData\\Roaming\\Python\\Python310\\site-packages\\py_eddy_tracker\\data\\GHRSST\\*.nc4', output='GHRSST/SST_weekly/out.nc', fortran=True, dimension='analysed_sst')