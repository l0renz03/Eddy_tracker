import numpy as np
import matplotlib.pyplot as plt
import math
#from mpl_toolkits.basemap import Basemap
#import geopy.distance 
from matplotlib.widgets import Slider, Button


'''import geopy.distance

coords_1 = (52.2296756, 21.0122287)
coords_2 = (52.406374, 16.9251681)

print(geopy.distance.geodesic(coords_1, coords_2).km)'''

#New Code

def HDdistance (mainEddy, candidates):
    closestEddyDist = 100000                    #Set a really high closestEddyDist to initialise the variable
    for i in range(len(candidates)):  
       # print(mainEddy[0][1], candidates[i][0])                                        #Loop through all of the candidates
        HDdist = math.sqrt((mainEddy[0][0]-candidates[i][0][0])**2 + (mainEddy[0][1]-candidates[i][0][1])**2)   #Find the distances for each candidate
        if HDdist < closestEddyDist:            #If a candidate is closer to the main eddy that the others, it becomes the closestEddy
            closestEddy = candidates [i]

    return closestEddy                          #The final closest eddy from the candidates is returned.



def check_eddies(mainEddy, newEddies, dead): 
    candidates = []
    mainEddy_long = (mainEddy[0])[1]

    mainEddy_lat = mainEddy[0][0]

    mainEddy_area = mainEddy[1]
    mainEddy_amp = mainEddy[2]

    if dead == True:
        n = 2
    else:
        n = 1
    for newEddy in newEddies:
        newEddy_long = newEddy[0][1] 
        newEddy_lat = newEddy[0][0]
        newEddy_area = newEddy[1] 
        newEddy_amp = newEddy[2]
        
        if (abs(mainEddy_long - newEddy_long) <= 3*n) and ((abs(mainEddy_lat - newEddy_lat) <= 3*n)): #1.35deg latitude = 150 km  
                                 #1.5 deg long = 150 km these values are temporary                              
    
            if ((newEddy_area >= 0.25*mainEddy_area) and (newEddy_area <= 2.75*mainEddy_area)):  #check if newEddy is in correct range for area
                if ((newEddy_amp >= 0.25*mainEddy_amp) and (newEddy_amp <= 2.75*mainEddy_amp)):  #check if newEddy is in correct range for amplitude
                    candidates.append(newEddy)       
                                      
    return candidates # returns list containing potential eddies






#ex_list = [[[(2, 2), 0.50, 5], [(42, 53), 10.0, 15], [(12, 18), 5.00, 4]], [[(2.2, 1.9), 0.51, 4.95], [(42.1, 52.8), 10.2, 15.1], [(11.95, 18.1), 4.99, 4.2]], [[(2.3, 2), 0.51, 4.95], [(42.2, 52.9), 10.2, 15.1]], [], [[(42.3, 52.8), 10.1, 15.2], [(2.35, 2.1), 0.52, 4.97], [(70.35, 90.1), 0.52, 4.97]]]


#Sample data
import random

random.seed(5)
num_weeks=52
eddy_no=100 
area_list = [[random.uniform(0.75*150, 3.75*150) for i in range(eddy_no)] for j in range(num_weeks)]

# 2. Generate random values between 0.25*2 and 2.75*2 for 7 weeks
amplitude_list = [[random.uniform(0.5*2, 3*2) for i in range(eddy_no)] for j in range(num_weeks)]

# 3. Generate 20 random points on a 100 by 100 grid for 7 weeks
coordinates_list = [[(random.uniform(-100, 0), random.uniform(0, 100)) for i in range(eddy_no)] for j in range(num_weeks)]
# Create a nested list where each item is a sublist containing the coordinate, area, and amplitude for each point
ex_list = [[[coordinates_list[i][j], area_list[i][j], amplitude_list[i][j]] for j in range(20)] for i in range(num_weeks)]
#print(ex_list)


search_range = 150 #Sample 
output = [] #Output initialization
current_week_counter = 1 #Variable

for week in ex_list:

    for eddy in week:
        eddy_list = [[eddy, current_week_counter]]               #list containing all position of specific eddy through time
        
        disappeared = False
        new_week_counter = current_week_counter + 1
        for new_week in ex_list[(current_week_counter):]:       #loop through all weeks to find position of eddy
                 
            main_eddy = eddy_list[-1][0] 
                                      #get last position of eddy that has been found for analysis
            candidates = check_eddies(main_eddy, new_week, disappeared)           #get candidates 
            
            if len(candidates) == 1:                                  #if only one candidate found, append
                eddy_list.append([candidates[0], new_week_counter])
                disappeared = False
                new_week.remove(candidates[0])                      #Delete eddy once it has been added
                #Delete eddy
              
            elif len(candidates) > 1:                                 #if more than one candidate, append the one with smallest distance
                closest_eddy = HDdistance(main_eddy, candidates)
                eddy_list.append([closest_eddy, new_week_counter])
                disappeared = False
                new_week.remove(closest_eddy)
                #Delete eddy
                
            elif len(candidates) == 0:
                if disappeared == True:
                    break
                else:
                    disappeared = True
            new_week_counter += 1   
        output.append(eddy_list)  
          
    current_week_counter += 1
    

outputFiltered = []             # Creates a list for the eddies that last for 5 weeks or longer
for m in output: 
    dif = m[-1][1] - m[0][1]                          # Iterates over the ouput list
    if dif >= 4:                      # Checks if the eddy lasts 5 or more weeks
        outputFiltered.append(m)        # If the eddy lasts 5 or more weeks, it is added to the filtered eddies
        
                
#for i in outputFiltered:
    #print("eddy",i)

#Counting stuff


eddies=[0]*52
new_eddies=[0]*52
dead_eddies=[0]*52
for eddy in outputFiltered:  # Look at each eddy in the filtered data
    first_week_n = eddy[0][1]
    last_week_n = eddy[-1][1]
    for i in range((first_week_n -1), last_week_n):              #Adds live eddy to each week
        eddies[i] = eddies[i] + 1
    if last_week_n <52:
        dead_eddies[last_week_n] = dead_eddies[last_week_n] +1   #Adds dead eddy

for i in range(1,len(dead_eddies)):                     #Adds up dead eddies
    dead_eddies[i] += dead_eddies[i-1]


    
x_cords=np.linspace(0,52,52)
plt.plot(x_cords,eddies)
plt.plot(x_cords,new_eddies)
plt.plot(x_cords,dead_eddies)
legends=['eddy','new','dead']
plt.legend(legends)


#for i in range(1,len(dead_eddies)):                     #Adds up dead eddies
#    dead_eddies[i] += dead_eddies[i-1]



# #Plotting 

# m = Basemap(projection='mill', llcrnrlat=0, urcrnrlat=80, llcrnrlon=-100, urcrnrlon=20)

# # Draw coastlines and boundaries
# m.drawcoastlines()
# m.drawcountries()
# m.fillcontinents(color='green', lake_color='white')

# for eddy in outputFiltered:
#     x_cords=[]
#     y_cords=[]
#     for week in eddy:
#         x = week[0][0][0]
#         y = week[0][0][1]
#         x,y=np.round(x,1),np.round(y,1)
#         x_cords.append(x)
#         y_cords.append(y)
#     xx,yy=m(x_cords, y_cords)
#     plt.plot(xx,yy,'-')
# plt.show()


# #plots circles for each week
# for week in range(num_weeks):    
#     fig, ax = plt.subplots()
#     for eddy in outputFiltered:
#         x_cords = []
#         y_cords = []
#         for part in eddy:
#             if part[1] == week:
#                 x = part[0][0][0]
#                 y = part[0][0][1]
#                 eddyCircle = plt.Circle((x, y), part[0][2], color="r", fill=False)
#                 x_cords.append(x)
#                 y_cords.append(y)

#                 ax.add_patch(eddyCircle)
#                 ax.plot(x_cords, y_cords)

#     fig.show()

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(figsize=(12, 12))


for eddy in outputFiltered:
    x_cords = []
    y_cords = []
    for week in eddy:
        cur_week = week[1]
        if cur_week > 1:
            break
        x = week[0][0][0]
        y = week[0][0][1]
        x, y = np.round(x, 1), np.round(y, 1)
        x_cords.append(x)
        y_cords.append(y)
    ax.plot(x_cords, y_cords, '-')
    plt.xlim([-100, 0])

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])

allowed_val = np.arange(1,53,1)

week_slider = Slider(
    ax=axfreq,
    label='Week',valmin =1,valmax =52,
    valinit=1,
    valstep=allowed_val
)

# The function to be called anytime a slider's value changes
def update(val):
    ax.clear()
    
    for eddy in outputFiltered:
        x_cords = []
        y_cords = []
        radius = 0
        for week in eddy:
            cur_week = week[1]
            if cur_week > week_slider.val:
                break
            x = week[0][0][0]
            y = week[0][0][1]
            x, y = np.round(x, 1), np.round(y, 1)
            x_cords.append(x)
            y_cords.append(y)
            radius = week[0][2]
            
        if bool(x_cords) == True:
            cur_eddies = eddies[week_slider.val-1]
            #cur_new_eddies = new_eddies[week_slider.val-1]
            cur_dead_eddies = dead_eddies[week_slider.val-1]
            ax.text(-100, -2,f'Current Eddies = {cur_eddies}\nDead Eddies = {cur_dead_eddies}', 
         style = 'italic',
         fontsize = 10,
         color = "green")
            if cur_week < week_slider.val:
                eddyCircle = plt.Circle((x_cords[-1], y_cords[-1]), radius, color="0.8", fill=False)
                ax.add_patch(eddyCircle)
                ax.plot(x_cords, y_cords, '-', color="0.8")
                ax.set_xlim(-80, 25)
                ax.set_ylim(0, 70)
            else:
                eddyCircle = plt.Circle((x_cords[-1], y_cords[-1]), radius, color="r", fill=False)
                ax.add_patch(eddyCircle)
                ax.plot(x_cords, y_cords, '-')
                ax.set_xlim(-80, 25)
                ax.set_ylim(0, 70)



# register the update function with each slider
week_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    week_slider.reset()
button.on_clicked(reset)

ax.set_xlim(-80, 25)
ax.set_ylim(0, 70)
plt.show()





'''
for eddy in outputFiltered:
    x_cords=[]
    y_cords=[]
    for week in eddy:
        x = week[0][0][0]
        y = week[0][0][1]
        x_cords.append(x)
        y_cords.append(y)
    plt.plot(x_cords,y_cords)
plt.xlabel("Latitude [°]")
plt.ylabel("Longitude [°]")
#ax = plt.gca()
#ax.set_facecolor('lightblue')
plt.savefig("filepath.png", format = 'png', dpi=1000)'''
