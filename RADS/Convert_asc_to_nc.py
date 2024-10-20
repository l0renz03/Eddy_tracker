"""
Take RADS data, grid it, convert to array, convert to .nc file
"""

import numpy as np
from numpy import ma
import netCDF4 as nc
import os
from netCDF4 import Dataset
#from global_land_mask import globe



# Load the .npz file
data = np.load('RADS\getRADSgridAllSat_memory\getRADSgridAllSat_1_7__-85_25_0_70.npz')

# Extract the arrays from the .npz file
sla = data['arr_0']
longitude = data['arr_1'][:,0].ravel()
latitude = data['arr_2'][0].ravel()

print(len(latitude))
print(len(longitude))
print(np.shape(sla))

# Create a new .nc4 file
output_dir = 'RADS/RADS_in_nc'
output_file = os.path.join(output_dir, 'output_file_1_43.nc4')
ncfile = Dataset(output_file, mode='w', format='NETCDF4')


# Define dimensions
ncfile.createDimension('time', None)
ncfile.createDimension('latitude', len(latitude))
ncfile.createDimension('longitude', len(longitude))

# Define variables
time_var = ncfile.createVariable('time', np.int32, ('time',))
lat_var = ncfile.createVariable('latitude', latitude.dtype, ('latitude',))
lon_var = ncfile.createVariable('longitude', longitude.dtype, ('longitude',))
sla_var = ncfile.createVariable('sla', sla.dtype, ('time', 'longitude', 'latitude',))

# Write data to variables
lat_var[:] = latitude
lon_var[:] = longitude
sla_var[0,:,:] = np.ma.masked_where(np.isnan(sla),sla)#sla # Assuming there's only one time step in the data
sla_var.set_auto_mask(False)
time_var[0] = 24294. # Assuming time starts at 0 and there's only one time step in the data

#overlay = globe.is_land(lat_for_globe, lon_for_globe)
# sla_var[overlay==True] = 0
#sla_var = np.ma.masked_array(sla_var, mask=(overlay == True))


# Close netCDF4 file
ncfile.close()

# # Load the contents of the .npz file into a dictionary
# npz_file = np.load('RADS\getRADSgridAllSat_memory\getRADSgridAllSat_1_7__-85_25_0_70.npz')
# data_dict = dict(npz_file.items())
# npz_file.close()

# # Create the output folder if it doesn't already exist
# output_folder = os.path.join('RADS', 'RADS_in_nc')
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Combine all arrays into a single multidimensional array
# data_combined = np.array(list(data_dict.values()))

# # Create a new .nc4 file for the combined data and write the data to it
# nc_file = nc.Dataset(os.path.join(output_folder, 'combined_data.nc'), 'w', format='NETCDF4_CLASSIC')
# # Create a NetCDF4 dimension object for each dimension of the combined data
# dims = [nc_file.createDimension('dim{}'.format(i+1), dimsize) for i, dimsize in enumerate(data_combined.shape)]
# # Create a NetCDF4 variable for the combined data
# nc_var = nc_file.createVariable('combined_data', data_combined.dtype, tuple(['dim{}'.format(i+1) for i in range(data_combined.ndim)]))
# # Write the data to the variable
# nc_var[:] = data_combined[:]
# nc_file.close()



# fn = 'RADS\RADS_in_netCDF'
# ds = nc.Dataset(fn, 'w', format = 'NETCDF4')
