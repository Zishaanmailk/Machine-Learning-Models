import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# feching dependent(y) and independent(x) varibles
data=pd.read_csv('data.csv')
print(data)
x=np.array(data['x'])
y=np.array(data['y'])

# declareing mean
xmean=np.mean(x)
ymean=np.mean(y)

# calculating numarator, dnomeator, slope and intercept
num=np.sum((x-xmean)*(y-ymean))
dno=np.sum((x-xmean)**2)
slope=num/dno
inter=ymean-(slope*xmean)

print("slope is ",slope)
print("intersept is ",inter)

# pridicting dependent valus
ypred=(slope*x)+inter

#ploting graph
plt.scatter(x,y,c="red",label="actual points")
plt.plot(x,ypred,label="pridection")
plt.title('linear regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

