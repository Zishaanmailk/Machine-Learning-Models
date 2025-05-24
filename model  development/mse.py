import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x=np.linspace(0,100,100)
y=np.linspace(0,100,100)
y=y+np.random.normal(1,6,100)
xm=np.mean(x)
ym=np.mean(y)
m=(np.sum((xm-x)*(ym-y)))/np.sum((xm-x)**2)
c=ym-m*xm
yper=m*x+c
data2={'x':x,'y':y,'yper':yper}
df=pd.DataFrame(data2)
cost=np.mean(y-yper)
df.to_csv("data2")
plt.plot(x,yper,c="red",label="predicted line")
plt.scatter(x,y,s=2,label="actual points")
plt.legend()
plt.show()
