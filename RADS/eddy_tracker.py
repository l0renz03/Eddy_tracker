import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.basemap import Basemap
import geopy.distance as geod 
from matplotlib.widgets import Slider, Button
import time
from matplotlib.colors import ListedColormap
# #import geopy.distance
print("Start")
start = time.time()


#Input data is (latitude,longitude)
#New Code

def HDdistance (mainEddy, candidates):
    closestEddyDist = 100000                    #Set a really high closestEddyDist to initialise the variable
    for i in range(len(candidates)):  
                                             #Loop through all of the candidates
        #HDdist = math.sqrt((mainEddy[0][0]-candidates[i][0][0])**2 + (mainEddy[0][1]-candidates[i][0][1])**2)   
        HDdist = geod.geodesic((mainEddy[0]),(candidates[i][0])).km        # #Find the distances for each candidate
        if HDdist < closestEddyDist:            #If a candidate is closer to the main eddy that the others, it becomes the closestEddy
            closestEddy = candidates [i]
       
    return closestEddy                          #The final closest eddy from the candidates is returned.
def eddyVelocity(eddyNum):
    vlist = []
    
    for i in range(len(outputFiltered[eddyNum])-1):


        eddyDist = geod.geodesic((outputFiltered[eddyNum][i][0][0]), (outputFiltered[eddyNum][i+1][0][0])).km
        
        eddytime=outputFiltered[eddyNum][i+1][1]-outputFiltered[eddyNum][i][1]
       
        eddyVel = eddyDist/(604.8*eddytime)
        week=outputFiltered[eddyNum][i][1]
        lifetime = outputFiltered[eddyNum][-1][1] - outputFiltered[eddyNum][0][1]
        vlist.append([eddyVel,week,lifetime])
        

    return vlist
def check_eddies(mainEddy, newEddies, dead): 
    candidates = []
    # mainEddy_long = (mainEddy[0])[1]

    # mainEddy_lat = mainEddy[0][0]

    mainEddy_area = mainEddy[1]
    mainEddy_amp = mainEddy[2]

    if dead == True:
        n = 2
    else:
        n = 1
    for newEddy in newEddies:
        # newEddy_long = newEddy[0][1] 
        # newEddy_lat = newEddy[0][0]
        newEddy_area = newEddy[1] 
        newEddy_amp = newEddy[2]
        dist = geod.geodesic((mainEddy[0]),(newEddy[0])).km
        #if (abs(mainEddy_long - newEddy_long) <= 4*n) and ((abs(mainEddy_lat - newEddy_lat) <= 4*n)): #1.35deg latitude = 150 km  
                                 #1.5 deg long = 150 km these values are temporary                              
        if dist < 150*n:
            if ((newEddy_area >= 0.25*mainEddy_area) and (newEddy_area <= 2.75*mainEddy_area)):  #check if newEddy is in correct range for area
                if ((newEddy_amp >= 0.25*mainEddy_amp) and (newEddy_amp <= 2.75*mainEddy_amp)):  #check if newEddy is in correct range for amplitude
                    candidates.append(newEddy) 
                  
                                      
    return candidates # returns list containing potential eddies

#ACTUAL DATA
import velocity_field_tracking 
#tracking_input.txt




startweek = 40
endweek = 52
def load_data(start_week,end_week):   # load data from txt file
    with open('tracking_inputnew.txt', 'r') as f:
        # Read in each line of the file as a string
            lines = f.readlines()

    import ast
    # Strip newline characters and create a list of the resulting strings
    ex_list = [ast.literal_eval(line.strip()) for line in lines][start_week:end_week]
    return ex_list






#To LOAD DATA FROM FUNCTION(uncomment to load)

#data = velocity_field_tracking.get_data_for_tracker(startweek,endweek)
#with open('tracking_inputnew.txt', 'w') as f:
#    for item in data:
#        f.write("%s\n" % item)
# ex_list=data


ex_list=load_data(startweek-1,endweek)   #TO LOAD DATA FROM LIST
 
print("Data loaded!")


#************************************************************************************************************************************************
output = [] #Output initialization
current_week_counter = 1 #Variable
print("Begininning for loop")
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
    print("week", (current_week_counter-1) , "done!. Analyzed", len(week), "eddies")

print("raw data obtained!")
outputFiltered = []             # Creates a list for the eddies that last for 5 weeks or longer
for m in output: 
    dif = m[-1][1] - m[0][1]                          # Iterates over the ouput list
    if dif >= 4:                      # Checks if the eddy lasts 5 or more weeks
        outputFiltered.append(m)        # If the eddy lasts 5 or more weeks, it is added to the filtered eddies
print("Filtering done!")

eddyVels = []
for i in range (len(outputFiltered)):    
    v = eddyVelocity(i)
    eddyVels.append(v)

#Velocity Plotting 
week_vels=[]
for t in range(1,52):   #Looks at weeks 1 to 51. 52 not there as no reference eddies for 53
    week_vel=[]
    for i in eddyVels:
        for eddytime in i:
            week_no = eddytime[1]
            if week_no==t:
                week_vel.append(eddytime[0])
    week_vels.append(week_vel)

avgVels = []

for week in week_vels:    #Calculates average velocity of eddies per week
    if not len(week)==0:
        avg = sum(week)/(len(week))
    else:
        avg=0
    avgVels.append(avg)



avgVelslife = []
for i in range (0, len(eddyVels)):  #Calculates average velocities of eddy at it's point of lifetime
    sumVel = 0
    count = 0
    for eddies in eddyVels[i]:
        if len(eddies) > i:
            sumVel += eddies[0]
            count += 1
    if count == 0:
        avgVel = 0
    else:
        avgVel = sumVel/count
    avgVelslife.append(avgVel)




#Lifetime in terms of %
vel_pct=[]
for eddy in eddyVels:
    if len(eddy)>15:
        den = eddy[-1][1] - eddy[0][1]
        init_week = eddy[0][1]
        vel_cur_pct = []
        for eacheddy in eddy:
            life_pct = (eacheddy[1]-init_week)/den
            velocity = eacheddy[0]
            #vel_cur_pct.append([velocity,life_pct])
            vel_pct.append([velocity,life_pct])
print(vel_pct)



#Average velocity for starting and ending week and mid life 




# eddies=[0]*52
# new_eddies=[0]*52
# dead_eddies=[0]*52
# for eddy in outputFiltered:  # Look at each eddy in the filtered data
#     first_week_n = eddy[0][1]
#     last_week_n = eddy[-1][1]
#     for i in range((first_week_n -1), last_week_n):              #Adds live eddy to each week
#         eddies[i] = eddies[i] + 1
#     if last_week_n <52:
#         dead_eddies[last_week_n] = dead_eddies[last_week_n] +1   #Adds dead eddy

# for i in range(1,len(dead_eddies)):                     #Adds up dead eddies
    #dead_eddies[i] += dead_eddies[i-1]


import os 
print("the number of eddies are", len(outputFiltered))
#Plotting
m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=70, llcrnrlon=-80, urcrnrlon=25)
folder = "RADS/Overleaf_images/Tracking_images"

def plot_onmap(condition): #Simple plot on a map
    if condition==True:
        m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=70, llcrnrlon=-80, urcrnrlon=25)

    # Draw coastlines and boundaries
    m.drawcoastlines()
    m.drawcountries()
    # m.fillcontinents(color='green', lake_color='white')
    for eddy in outputFiltered:
        x_cords=[]
        y_cords=[]
        for week in eddy:
            x = week[0][0][0]
            y = week[0][0][1]
            x_cords.append(x)
            y_cords.append(y)
        x,y = m(y_cords, x_cords)
        m.plot(x,y)
        plt.xlabel("Longitude [°]")
        plt.ylabel("Latitude [°]")
        plt.savefig("eddy_map.png")
        plt.savefig("eddy_map.svg")
        plt.show()

def plot_based_on_age(condition): #Plots on a map and shows gradient colour changing as eddy ages
    if condition==True:
        fig, ax = plt.subplots(figsize=(10, 10))
        m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=70, llcrnrlon=-85, urcrnrlon=10)

        # Draw coastlines and boundaries
        m.drawcoastlines(linewidth=1)
        m.drawcountries()
        m.fillcontinents(color='green', lake_color='white')
        m.drawparallels(np.arange(25,70,10),labels=[1,1,0,1], fontsize=14)
        m.drawmeridians(np.arange(-85,10,10),labels=[1,1,0,1], rotation=45, fontsize=14)

        num_colors = 10000  # choose the number of colors in the colormap
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_colors))
        cmap = ListedColormap(colors)

       
        
        for eddy in outputFiltered:
            x_cords=[]
            y_cords=[]
            for week in eddy:
                x = week[0][0][0]
                y = week[0][0][1]
                x_cords.append(x)
                y_cords.append(y)
            x,y = m(y_cords, x_cords)
            
            # calculate the position along the track for each point
            track_length = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            track_length = np.insert(track_length, 0, 0)  # prepend a zero to match the length of x and y
            track_pos = track_length / np.max(track_length)
            
            # look up the color in the colormap based on the position along the track
            colors = cmap(track_pos)
        
            # plot the lines with the color varying over position
            for i in range(len(x)-1):
                m.plot([x[i], x[i+1]], [y[i], y[i+1]], color=colors[i], linewidth=2)


        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        cbar = plt.colorbar(sm, ax=ax,fraction=0.03,pad=0.12)
        cbar.set_label('Age fraction', fontsize=18)
        
        filename = f"{startweek} to {endweek} weeks_age.png"
        filename1 = f"{startweek} to {endweek} weeks_age.svg"
        # plt.title("Position of eddies with map overlay")
        plt.xlabel("Longitude [°]", fontsize=18,labelpad = 45)
        plt.ylabel("Latitude [°]",fontsize=18, labelpad = 45)
        plt.savefig(os.path.join(folder,filename),bbox_inches='tight')
        plt.savefig(os.path.join(folder,filename1),bbox_inches='tight')
        #plt.show()
def plot_vels(condition):  #Plots average velocities over weeks
    if condition==True:
        x_axis = np.arange(1,len(avgVels)+1,1)
        plt.plot(x_axis,avgVels)
        plt.title("Velocity of eddies (m/s) against weeks ")
        plt.xlabel("Week")
        plt.ylabel('Average velocity [m/s]')
        plt.savefig("speed.png")
        plt.savefig("speed.svg")
        plt.show()
def plot_vels_based_on_life(condition):
    if condition==True:
        x_axis = np.arange(1,52,1)
        plt.plot(x_axis,avgVelslife)
        plt.title("Velocity of eddies (m/s) against lifetime  ")
        plt.savefig("speed_life.png")
        plt.savefig("speed_life.svg")
        plt.show()

def plot_based_on_percentage_of_life_squre():  
    x_cords =[]
    y_cords=[]
    for eddy in vel_pct:        
        #for vel in eddy:
        # x_cords.append(vel[1]*100)
        # y_cords.append(vel[0]**2)
        x_cords.append(eddy[1]*100)
        y_cords.append(eddy[0]**2)
    plt.plot(x_cords,y_cords)
    plt.savefig("speed_pctlife_2.png")
    plt.savefig("speed_pctlife_2.svg")
    plt.xlabel("Lifetime percentage [%]")
    plt.ylabel('Velocity^2 [m/s]')
    plt.show()
def plot_based_on_percentage_of_life():
    x =[]
    y=[]
    #vel_pct=np.array(vel_pct)
    #vel_norm = (vel_pct-np.min(vel_pct))/(np.max(vel_pct)-np.min(vel_pct))
    for eddy in vel_pct:        
        #for vel in eddy:
        x.append(eddy[1]*100)
        y.append(eddy[0])
    y=np.array(y)
    y_norm = (y-np.min(y))/(np.max(y)-np.min(y))
    plt.scatter(x,y_norm)
    plt.savefig("speed_pctlife.png")
    plt.savefig("speed_pctlife.svg")
    plt.xlabel("Lifetime percentage [%]")
    plt.ylabel('Velocity [m/s]')
    plt.show()

def plot_vels_basedonweek():
    x_cords = np.arange(0,len(new_eddy_velocities),1)
    y_cords=new_eddy_velocities
    plt.plot(x_cords,y_cords)
    plt.title("new week velocities")
    plt.show()


def plot_based_on_age_smol(condition): #Plots on a map and shows gradient colour changing as eddy ages
    if condition==True:
        fig, ax = plt.subplots(figsize=(10, 10))
        m = Basemap(projection='merc', llcrnrlat=28, urcrnrlat=60, llcrnrlon=-85, urcrnrlon=-10)

        # Draw coastlines and boundaries
        m.drawcoastlines(linewidth=1)
        m.drawcountries()
        m.fillcontinents(color='green', lake_color='white')
        m.drawparallels(np.arange(28,60,10),labels=[1,1,0,1], fontsize=14)
        m.drawmeridians(np.arange(-85,-10,10),labels=[1,1,0,1], rotation=45, fontsize=14)

        num_colors = 10000  # choose the number of colors in the colormap
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_colors))
        cmap = ListedColormap(colors)
 
       
        
        for eddy in outputFiltered:
            x_cords=[]
            y_cords=[]
            for week in eddy:
                x = week[0][0][0]
                y = week[0][0][1]
                x_cords.append(x)
                y_cords.append(y)
            x,y = m(y_cords, x_cords)
            
            # calculate the position along the track for each point
            track_length = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            track_length = np.insert(track_length, 0, 0)  # prepend a zero to match the length of x and y
            track_pos = track_length / np.max(track_length)
            
            # look up the color in the colormap based on the position along the track
            colors = cmap(track_pos)
        
            # plot the lines with the color varying over position
            for i in range(len(x)-1):
                m.plot([x[i], x[i+1]], [y[i], y[i+1]], color=colors[i], linewidth=2)


        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        cbar = plt.colorbar(sm, ax=ax,fraction=0.03,pad=0.12)
        cbar.set_label('Age fraction', fontsize=18)

        filename = f"{startweek} to {endweek} weeks_age_small.png"
        filename1 = f"{startweek} to {endweek} weeks_age_small.svg"
        # plt.title("Position of eddies with map overlay")
        plt.xlabel("Longitude [°]",fontsize=18, labelpad = 45)
        plt.ylabel("Latitude [°]", fontsize=18,labelpad = 45)
        plt.savefig(os.path.join(folder,filename),bbox_inches='tight')
        plt.savefig(os.path.join(folder,filename1),bbox_inches='tight')
        #plt.show()



#Call functions

plot_based_on_age(True)
plot_based_on_age_smol(True)




print("time needed to run code is:", (time.time() - start)/60 , "min")



















