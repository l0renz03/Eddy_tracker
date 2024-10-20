# imports
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata

from rads_funcs import grid_data, data_plot


# directory of data
filedir = 'RADS/example/'

X = np.array([])
Y = np.array([])
T = np.array([])


fig, ax = plt.subplots(1, 2)

for file in os.listdir(filedir): # cycle through directory
    if file.endswith('.asc'): # pick only data files
        example_data = np.loadtxt(filedir+file, skiprows=13) # convert data to numpy array
        if example_data.ndim == 2: # pick only arrays with multiple data points
            example_data = example_data[~np.isnan(example_data)[:, 3], :] # removing NaN data points
            #x, y = m(example_data[:, 2], example_data[:, 1]) # converting coordinates for plotting
            x, y = (example_data[:, 2], example_data[:, 1])
            t = example_data[:, 3]
            X = np.concatenate((X, x))
            Y = np.concatenate((Y, y))
            T = np.concatenate((T, t))



data_plot((X, Y, T), 'Interpolated Data', ax[1], llcrnrlat=0, urcrnrlat=70, llcrnrlon=-85, urcrnrlon=25)


# normalization of the colormap, min=-1m max=+1m
norm = mpl.colors.Normalize(-1, 1)

m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=70,
            llcrnrlon=-85, urcrnrlon=25, lat_ts=20, resolution='l', ax=ax[0])
ax[0].set_title('Raw Data')
x, y = m(X, Y)
ax[0].scatter(x, y, s=np.ones(x.shape)/100, c=T, cmap='seismic', norm=norm)

m.drawcoastlines()
m.fillcontinents()
# draw parallels
m.drawparallels(np.arange(10,90,20),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(-180,180,30),labels=[1,1,0,1])

# plot colorbar
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='seismic'), ax=ax, label='Sea Level Anomaly [m]', orientation='horizontal') #, fraction=0.02, pad=0.04)

#plt.tight_layout()
fig.savefig('example_rads/raw_interp_comp.png', dpi=600, bbox_inches='tight')
