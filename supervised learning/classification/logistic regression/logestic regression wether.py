import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data2 = pd.read_csv("weather.csv")
data2.isna()
print(len(data2))
nulcount = data2.isnull().sum().sum()
print(nulcount)
data2 = data2.dropna(axis=0)
data2 = data2.reset_index(drop=True)
print(len(data2))
nulcount = data2.isnull().sum().sum()
print(nulcount)

l = pd.DataFrame()
l["WindGustDir"] = data2.WindGustDir
l["WindDir9am"] = data2.WindDir9am
l["WindDir3pm"] = data2.WindDir3pm


l1 = LabelEncoder()
l2 = LabelEncoder()
l3 = LabelEncoder()
l["WindGustDir"] = l1.fit_transform(l["WindGustDir"])
l["WindDir9am"] = l2.fit_transform(l["WindDir9am"])
l["WindDir3pm"] = l3.fit_transform(l["WindDir3pm"])

inputt = data2.drop(["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow"], axis=1)
rt = pd.DataFrame()
l4 = LabelEncoder()
rt["RainToday"] = l4.fit_transform(data2["RainToday"])
print(rt)

inputt["RainToday"] = rt["RainToday"]
inputt["WindGustDir"] = l["WindGustDir"]
inputt["WindDir9am"] = l["WindDir9am"]
inputt["WindDir3pm"] = l["WindDir3pm"]
print(inputt)


inputt.to_csv("inputt.csv", index=False)

target = pd.DataFrame()
target["RainTomorrow"] = data2.RainTomorrow
print(target)

l5 = LabelEncoder()
target["RainTomorrow"] = l5.fit_transform(target["RainTomorrow"])
print(target)

xxx, xxt, yy, yyt = train_test_split(inputt, target, test_size=0.1)


model2 = LogisticRegression(max_iter=2000)
model2.fit(xxx, yy.values.ravel()) #convert 1d for y

ppy = pd.DataFrame()
ppy["RainTomorrow"] = model2.predict(xxt)
print(ppy)


print("Accuracy:", model2.score(xxt, yyt))

sns.heatmap(inputt.corr(), annot=True)
plt.show()
sns.heatmap(inputt.corr())
plt.show()
yc = confusion_matrix(yyt, ppy)
sns.heatmap(yc, annot=True)


plt.xlabel("truth")
plt.ylabel("predicted")
plt.show()
