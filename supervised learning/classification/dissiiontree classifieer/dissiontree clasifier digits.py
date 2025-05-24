import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
data=load_digits()
print(dir(data))
x=data.data
y=data.target
plt.gray()
for i in range(5):
    plt.matshow(data.images[i])
x, tx, y, ty = train_test_split(x, y, test_size=0.2)
model=DecisionTreeClassifier()
model.fit(x,y)
py=model.predict(tx)
print(x)
print(y)
print(tx)
print(ty)
print(py)
cm=confusion_matrix(ty,py)
sns.heatmap(cm,annot=True)
plt.xlabel("truth")
plt.ylabel("pridected")
print(model.score(tx,ty))
plt.show()