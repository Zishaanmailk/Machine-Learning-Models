import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("data2")
x=np.array(data['x'])
y=np.array(data['y'])

def train(x,y):
    xm=np.mean(x)
    ym=np.mean(y)
    m=(np.sum((xm-x)*(ym-y)))/np.sum((xm-x)**2)
    c=y-m*xm
    yper=m*x+c
    print("mean of x: ",xm)
    print("mean of y: ", ym)
    return yper

def plot2d(x,y,yper):
    plt.plot(x,yper,c="red",label="predicted line")
    plt.scatter(x,y,s=2,label="actual points")
    plt.legend()
    plt.show()



def plot3d(x,y,yper):
    # plotting
    z = np.linspace(1, 100, 100)
    ax = plt.axes(projection="3d")
    ax.scatter(x, y, z)
    xx, yyperr = np.meshgrid(x, yper)
    zz = xx + yyperr
    ax.plot_surface(xx, zz, yyperr)
    plt.pause(0.001)
    plt.show()


def grd(x,y):
   mc=c=0
   iter=100
   n=len(x)
   lern=0.00001
   plt.scatter(x, y,c="blue", label="actual points")
   plt.legend()
   for i in range(iter):
       yperr=mc*x+c
       #partial differentation of MSE
       md=-(2/n)*sum(x*(y-yperr))
       cd=-(2/n)*sum(y-yperr)
       mse = np.mean(y - yperr)
       if mse > 0:
           mc-=lern*md
           c-=lern*cd
       else:
           mc += lern * md
           c += lern * cd

       print(f"c= {c} m= {mc} itter= {i} cost= {mse}")
       plt.plot(x, yperr, c="red", label="predicted line")
       plt.title("Gradient Descent")
       plt.xlabel("Feature")
       plt.ylabel("Dependent value")
       plt.pause(0.01)

   plt.show()


def grd2(x,y):
   mc=c=0
   iter=100
   n=len(x)
   lern=0.00001

   for i in range(iter):
       yperr=mc*x+c
       #partial differentation of MSE
       md=-(2/n)*sum(x*(y-yperr))
       cd=-(2/n)*sum(y-yperr)
       mse = np.mean(y - yperr)
       if mse > 0:
           mc-=lern*md
           c-=lern*cd
       else:
           mc += lern * md
           c += lern * cd

       plt.cla()
       plt.scatter(x, y, c="blue", label="actual points")
       print(f"c= {c} m= {mc} itter= {i} cost= {mse}")
       plt.plot(x, yperr, c="red", label="predicted line")
       plt.title("Gradient Descent")
       plt.xlabel("Feature")
       plt.ylabel("Dependent value")
       plt.pause(0.01)
       plt.legend()

   plt.show()

#per=train(x,y)
#plot2d(x,y,per)

grd2(x,y)
grd(x,y)


