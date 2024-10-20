from global_land_mask import globe

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def landpoints():
    lon, lat, time = np.meshgrid(np.linspace(-85, 25, 85+25+1), np.linspace(0, 70, 71), np.linspace(0, 24*3600*7/(2**19), 8), indexing = 'ij')



    globemask = globe.is_land(lat, lon)

    lat, lon, time = lat.reshape(lat.size,1), lon.reshape(lon.size,1), time.reshape(time.size, 1)

    globemask = globemask.reshape(globemask.size,1)

    #print(lat.shape, lon.shape, globemask.shape)
    dimlist = [np.zeros([lon[globemask].size]), lat[globemask], lon[globemask], time[globemask]]

    extrapoints = np.stack(dimlist, axis = 1)
    
    return extrapoints


if __name__ == "__main__":
    #print(np.linspace(0, 24*3600*7, 8))

    #points = landpoints()

    #np.save("RADS/land0points.npy", points)

    #'''

    points = np.load("RADS/land0points.npy")
    fig, ax = plt.subplots()

    # initiating basemap
    m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=70, llcrnrlon=-85, urcrnrlon=25, lat_ts=20, resolution='l', ax=ax)
    
    m.drawcoastlines()

    lon, lat = m(points[:, 2], points[:, 1])

    m.scatter(lon, lat, s=np.ones((points.shape[0],))/100)
    plt.show()
    print(points)
    #'''