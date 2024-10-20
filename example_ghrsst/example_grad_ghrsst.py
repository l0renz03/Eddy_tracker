import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

f = Dataset('GHRSST/example/20230101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc')

lat = f['lat'][:]
lon = f['lon'][:]

lat0 = np.where(lat == 30)[0][0]
lat1 = np.where(lat == 50)[0][0] + 1

lon0 = np.where(lon == -80)[0][0]
lon1 = np.where(lon == -40)[0][0] + 1

sst_anomaly = f['sst_anomaly']
sst_anomaly = np.reshape(sst_anomaly[:,lat0:lat1,lon0:lon1], (lat1-lat0,lon1-lon0))

# normalization of the colormap, min=-10° max=+10°
normalize = mpl.colors.Normalize(-10, 10)

# initialization of the base map, Mercalli lat=[0, 70] lon [-85, 25] (North Atlantic Ocean in RADS)
m = Basemap(projection='merc',llcrnrlat=0,urcrnrlat=70,\
            llcrnrlon=-85,urcrnrlon=25,lat_ts=20,resolution='l')

lonm0, latm0 = m(-85,0)
lonm1, latm1 = m(25,70)

fig, ax = plt.subplots(2, 1)

ax[0].imshow(np.flip(sst_anomaly, axis=0), cmap='seismic', norm=normalize, interpolation='nearest', extent=[lon[lon0],lon[lon1],lat[lat0],lat[lat1]])
ax[1].contour(sst_anomaly, norm=normalize, cmap='seismic', extent=[lon[lon0],lon[lon1],lat[lat0],lat[lat1]])

fig.savefig('example_ghrsst/example_grad_ghrsst.png')
