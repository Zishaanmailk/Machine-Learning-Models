import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression ,LinearRegression ,Lasso ,Ridge
from sklearn.ensemble import RandomForestClassifier ,BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split ,KFold ,cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB ,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data2=pd.read_csv("weather.csv")
print("Loaded dataset:")
print(data2)
print("Total missing values:", data2.isnull().sum().sum())
print("Missing values per column:\n", data2.isnull().sum())
print("Data types:\n", data2.dtypes)

# Label encoding
lben=LabelEncoder()
data2['WindGustDir']=lben.fit_transform(data2['WindGustDir'])
data2["WindDir9am"]=lben.fit_transform(data2["WindDir9am"])
data2["WindDir3pm"]=lben.fit_transform(data2["WindDir3pm"])
data2["RainTomorrow"]=lben.fit_transform(data2["RainTomorrow"])
data2["RainToday"]=lben.fit_transform(data2["RainToday"])
print("After label encoding:\n", data2)

# Imputation
si=SimpleImputer(strategy='mean')
data2['Sunshine']=si.fit_transform(data2[['Sunshine']])
data2['WindGustSpeed']=si.fit_transform(data2[['WindGustSpeed']])
data2['WindSpeed9am']=si.fit_transform(data2[['WindSpeed9am']])
data2['WindGustDir']=si.fit_transform(data2[['WindGustDir']])
print("Total missing values after imputation:", data2.isnull().sum().sum())


inputt=data2.drop(["RainTomorrow"],axis=1)
targett=pd.DataFrame()
targett['RainToday']=data2['RainToday']
print("Target column:\n", targett)

# Models and cross-validation
print("Decision Tree:")
dt=cross_val_score(DecisionTreeClassifier(),inputt,targett,cv=7)
print(dt)

print("Logistic Regression:")
log=cross_val_score(LogisticRegression(max_iter=5000),inputt,targett,cv=7)
print(log)

print("SVM:")
svm=cross_val_score(SVC(),inputt,targett,cv=7)
print(svm)

print("Gaussian NB:")
gn=cross_val_score(GaussianNB(),inputt,targett,cv=7)
print(gn)

print("KNN:")
knn=cross_val_score(KNeighborsClassifier(n_neighbors=5),inputt,targett,cv=7)
print(knn)

print("Random Forest:")
rm=cross_val_score(RandomForestClassifier(n_estimators=100),inputt,targett,cv=7)
print(rm)

print("Bagging with Logistic Regression:")
bagging_model = BaggingClassifier(
    estimator=make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    n_estimators=100,
    random_state=42
)

#cross-validation
bag = cross_val_score(bagging_model, inputt, targett.values.ravel(), cv=7)

print(bag)


#bag=cross_val_score(
 #   BaggingClassifier(estimator=LogisticRegression(max_iter=5000), n_estimators=100, random_state=42),
   # inputt,
    #targett.values.ravel(),
    #cv=7
#)
#print(bag)
