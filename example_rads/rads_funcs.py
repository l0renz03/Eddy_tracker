# imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata


### Interpolate RADS data onto regular grid
#
## Inputs:
# data: (X, Y, T) are numpy arrays of the same size containing longitude, latitude and sla values respectively
## Keyword arguments:
# grid_resolution: resolution of the grid in degrees
# interpolation_method: method of interpolation to get grid points
# lat, lon: extents of the interpolation region

def grid_data(data, grid_resolution=0.1, interpolation_method='linear', lat_0=0, lat_1=70, lon_0=-85, lon_1=25):
    
    X, Y, T = data

    resx = int((lon_1 - lon_0) / grid_resolution) + 1
    resy = int((lat_1 - lat_0) / grid_resolution) + 1
    res = resx * resy

    x_grid = np.linspace(lon_0, lon_1, resx)
    y_grid = np.linspace(lat_0, lat_1, resy)

    grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing='ij')

    grid = griddata((X, Y), T, (grid_x, grid_y), method=interpolation_method)

    xx = grid_x.reshape((res,))
    yy = grid_y.reshape((res,))
    tt = grid.reshape((res,))

    return (xx, yy, tt)


### Plot RADS data
#
## Inputs:
# data: (X, Y, T) are numpy arrays of the same size containing longitude, latitude and sla values respectively
# title: the tile of the output plot
# axis: the axis on which the plot is plotted
## Note
# you need to initiate the plot before calling the function, for example
# fig, ax = plot.subplots()

def data_plot(data, title:str, axis, projection='merc', llcrnrlat=0, urcrnrlat=70,
            llcrnrlon=-85, urcrnrlon=25, lat_ts=20, resolution='l', coastlines=True, continents=True, grid_plot=True):

    xx, yy, tt = grid_data(data, lat_0=llcrnrlat, lat_1=urcrnrlat, lon_0=llcrnrlon, lon_1=urcrnrlon)
    
    axis.set_title(title)
    
    m = Basemap(projection=projection, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
            llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, lat_ts=lat_ts, resolution=resolution, ax=axis)

    xx, yy = m(xx, yy)

    # normalization of the colormap, min=-1m max=+1m
    normalize = mpl.colors.Normalize(-1, 1)

    axis.scatter(xx, yy, c=tt, cmap='seismic', norm=normalize)

    if coastlines:
        m.drawcoastlines()
    if continents:
        m.fillcontinents()
    if grid_plot:
        # draw parallels
        m.drawparallels(np.arange(10,90,20),labels=[1,1,0,1])
        # draw meridians
        m.drawmeridians(np.arange(-180,180,30),labels=[1,1,0,1])
