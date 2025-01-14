from netCDF4 import Dataset
import numpy as np
import os

'''
#f = Dataset('GHRSST/example/20230102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc4')
#f = Dataset('https://opendap.earthdata.nasa.gov/collections/C1996881146-POCLOUD/granules/20230102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.dap.nc4?dap4.ce=/mask%5B0:1:0%5D%5B8999:1:15999%5D%5B9499:1:20499%5D;/analysed_sst%5B0:1:0%5D%5B8999:1:15999%5D%5B9499:1:20499%5D;/lon%5B9499:1:20499%5D;/time%5B0:1:0%5D;/sea_ice_fraction%5B0:1:0%5D%5B8999:1:15999%5D%5B9499:1:20499%5D;/dt_1km_data%5B0:1:0%5D%5B8999:1:15999%5D%5B9499:1:20499%5D;/lat%5B8999:1:15999%5D;/analysis_error%5B0:1:0%5D%5B8999:1:15999%5D%5B9499:1:20499%5D;/sst_anomaly%5B0:1:0%5D%5B8999:1:15999%5D%5B9499:1:20499%5D')

#This is attempt push

print(f)

time = f['time']

lat = f['lat'][:]
lon = f['lon'][:]

print(lat, lon)

'''

MM = '12'

folder = 'C:/Users/User/OneDrive - Delft University of Technology/GHRSST_data/'+MM

for file in os.listdir(folder):
    date = file[0:8]

    try:

        #f = Dataset('GHRSST/example/20230101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc')
        #f = Dataset('GHRSST/example/20230102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc4')
        f = Dataset(folder+'/'+file)

        time = f['time']

        lat = f['lat'][:]
        lon = f['lon'][:]

        lat0 = np.where(lat == 30)[0][0]
        lat1 = np.where(lat == 50)[0][0] + 1

        lon0 = np.where(lon == -80)[0][0]
        lon1 = np.where(lon == -40)[0][0] + 1

        sst_anomaly = f['sst_anomaly']
        sst_anomaly = np.reshape(sst_anomaly[:,lat0:lat1,lon0:lon1], (lat1-lat0,lon1-lon0))
        


        ### PLOTTING ###
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap

        # normalization of the colormap, min=-10° max=+10°
        normalize = mpl.colors.Normalize(-10, 10)

        # initialization of the base map, Mercalli lat=[0, 70] lon [-85, 25] (North Atlantic Ocean in RADS)
        m = Basemap(projection='merc',llcrnrlat=0,urcrnrlat=70,\
                    llcrnrlon=-85,urcrnrlon=25,lat_ts=20,resolution='l')

        lonm0, latm0 = m(-85,0)
        lonm1, latm1 = m(25,70)

        fig, ax = plt.subplots()

        #m.drawcoastlines()
        #m.fillcontinents()
        # draw parallels
        #m.drawparallels(np.arange(10,90,20),labels=[1,1,0,1])
        # draw meridians
        #m.drawmeridians(np.arange(-180,180,30),labels=[1,1,0,1])

        ax.imshow(np.flip(sst_anomaly, axis=0), cmap='seismic', norm=normalize, interpolation='nearest', extent=[lon[lon0],lon[lon1],lat[lat0],lat[lat1]])
        ax.contour(sst_anomaly, norm=normalize, cmap='seismic', extent=[lon[lon0],lon[lon1],lat[lat0],lat[lat1]])
        ax.set_title(date)
        fig.savefig(f'example_ghrsst/images/{MM}/{date}.png')

        plt.close()

        print('DONE:', date)

    except:
        print('ERROR: There is no file for', date)
