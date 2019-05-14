import numpy as np
import imp
#import plotly.plotly as py
#import plotly.graph_objs as go



import numpy as np
# from plotly.graph_objs import *
import matplotlib

# import plotly.offline as py
# py.init_notebook_mode(connected=False)


matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_weight(cant):
    return   np.random.randint(3,size=(1,cant)) -1

def training(inputs):
    pos_desired = len(inputs[0]) - 1
    weight = get_weight(len(inputs[0]))[0]
    learn_rate=0.1
    change = True
    while (change):
        change = False
        for inp in inputs:
            y = inp[:len(inp)-1]
            y.append(1)
            result = np.dot(y,weight)
            if result< 0 and inp[pos_desired] is 1:       
                change = True
                y = np.array(y) * learn_rate
                weight = np.add(weight,y)
            elif result >= 0 and inp[pos_desired] is 0:
                change = True
                y = np.array(y) * learn_rate
                weight = np.subtract(weight,y)
    return(weight)       

entradas=[
    [0,0,0,0],
    [0,0,1,0],
    [0,1,0,1],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,1,0],
    [1,1,0,1],
    [1,1,1,1] 
]
entradas=[
    [0,0,0,0],
    [0,0,1,1],
    [0,1,0,1],
    [0,1,1,1],
    [1,0,0,1],
    [1,0,1,1],
    [1,1,0,1],
    [1,1,1,1] 
]



result = training(entradas) 
print("vector de pesos:",result)



fig = plt.figure()
ax = plt.axes(projection='3d')

x, y = np.meshgrid([-0.5, 1.5], [-0.5, 1.5])
z = -1/float(result[2])*(result[0]*x + result[1]*y + result[3])
ax.plot_surface(x, y, z, alpha=0.2)


for value, color in [(0, 'r'), (1, 'g')]: 
	xs = [p[0] for p in entradas if p[3] == value]
	ys = [p[1] for p in entradas if p[3] == value]
	zs = [p[2] for p in entradas if p[3] == value]

	ax.scatter(xs, ys, zs, color=color)


plt.show()