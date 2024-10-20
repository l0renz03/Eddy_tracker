'''File for Eddy Static Analysis'''

from rads_funcs import *
import time

# directory of data
filedir = 'RADS/example/'
#filedir = 'RADS/example/'

if __name__ == "__main__":

        extents = (100, 420, 50, 750) # Entire Grid: extents = (100, 420, 50, 750) --> 60;28 N -80;-10 W
                                      # Small grid (for method) extents = (330, 360, 250, 300)

        week = 'OW_paramter_choice'
        sla_grid, lon_grid, lat_grid = data_grid_selection(extents, 1, 7, plot=False, week=week)


        #Create folder for imgages:
        folder_to_create = folder_creation + f'Overleaf_images/week{week}'
        os.makedirs(folder_to_create, exist_ok=True)

        ow = 0.8 #Then negative is taken later
        #this code still sucks

        one_group = sla_pipeline(sla_grid, lon_grid, lat_grid, ow_cut= ow, uv_plot = False, ow_plot=True, plot_grouping=False, week = week)

        depths_array, depths_array_filter = depth_grid(depth_path, extents)
        output = eddy_pipeline(one_group, sla_grid, lat_grid, lon_grid, depths_array_filter, plot = False)
        print(len(output))



        print("FINISHED")

