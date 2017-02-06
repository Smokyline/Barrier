from alghTools.drawMap import *
from alghTools.tools import *
import numpy as np
direct = 'C:\\Users\\smoky\\Documents\\workspace\\result\\Barrier\\sample\\'

Xcoord = read_csv('C:\\Users\\smoky\\Documents\workspace\\resourses\\csv\\newEPA\\Caucasus_Xcoord.csv', ['x', 'y']).T
Vcoord = read_csv('C:\\Users\\smoky\\Documents\workspace\\resourses\\csv\\newEPA\\Caucasus_Vcoord.csv', ['x', 'y']).T
Allcoord = np.append(Xcoord, Vcoord, axis=0)
visuaMSu(Allcoord, Xcoord, Vcoord, r=0.225, title='diss', dir=direct, visual=True)
