from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata
from rads_funcs import get_data
from RADSgetgrid import RADSgrid
import matplotlib as mpl

from mpl_toolkits.basemap import Basemap

import numpy as np
import matplotlib.pyplot as plt

import time

t = [time.time()]

path = 'C:/Users/User/OneDrive - Delft University of Technology/RADS_data/CRYOSAT2'

RADS_grid = RADSgrid(path)

data = get_data(RADS_grid, 1, 7)

data = data[~np.isnan(data).any(axis=1)]

grid_resolution = 1

lon_0 = -85
lon_1 = 25
lat_0 = 0
lat_1 = 70

T = data[:, 0]/(3600*24)
X = data[:, 2]
Y = data[:, 1]
H = data[:, 3]

resx = int((lon_1 - lon_0) / grid_resolution) + 1
resy = int((lat_1 - lat_0) / grid_resolution) + 1
res = resx * resy

ttttt = int(7)

t_grid = np.linspace(T.min(), T.max(), ttttt)

x_grid = np.linspace(lon_0, lon_1, resx)
y_grid = np.linspace(lat_0, lat_1, resy)

grid_x, grid_y, grid_t = np.meshgrid(x_grid, y_grid, t_grid, indexing='ij')

res = grid_x.size

m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=70,
            llcrnrlon=-85, urcrnrlon=25, lat_ts=20, resolution='l')

xx = grid_x.reshape((res,))
yy = grid_y.reshape((res,))

xx, yy = m(xx, yy)

#interp = NearestNDInterpolator((X, Y, T), H)

#grid = interp(grid_x, grid_y, grid_t)

grid = griddata((X, Y, T), H, (grid_x, grid_y, grid_t), method='linear')

t.append(time.time())

print(f'DataGet: {t[-1]-t[0]} s')

plot = True

if plot:
    # normalization of the colormap, min=-1m max=+1m
    norm = mpl.colors.Normalize(-1, 1)

    for i in range(ttttt):
        grid_t = grid[:, :, i]



        fig, ax = plt.subplots()

        m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=70,
            llcrnrlon=-85, urcrnrlon=25, lat_ts=20, resolution='l', ax=ax)

        ax.set_title(i)
        #ax.scatter(xx, yy, c=)
        #ax.imshow(grid_t, cmap='seismic', norm=norm)
        m.coastlines()
        fig.savefig(f"RADS/TimeGrid/time_grid_{i:003}.png")
        plt.close()
        t.append(time.time())
        if i%10 == 0:
            print(f'{i}: {t[i+1]-t[i]} s')

#print(grid)
#print(grid.shape)

mean = False

if mean:
    grid = np.nanmean(grid, axis=2)


    print(grid.shape)
    print(grid)

    plt.imshow(grid)
    plt.show()





t.append(time.time())

print(f'DONE! {t[-1]-t[0]} s')



#grid = grid[:, :, 84]

#grid = grid.flatten(axis=2)


#print(grid)
#print(grid.shape)

#plt.imshow(grid)
#plt.show()

#np.savetxt('RADS/TimeGrid.txt', grid)





