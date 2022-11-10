import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
import pickle


data=pd.read_csv("healthcare-dataset-stroke-data.csv")
data=data.drop("id",axis=1)
data=data.dropna()

def outliersCapping(c):
    upper_limit = data[c].mean() + 3 * data[c].std()
    lower_limit = data[c].mean() - 3 * data[c].std()

    data[c] = np.where(
        data[c] > upper_limit,
        upper_limit,
        np.where(data[c] < lower_limit, lower_limit, data[c]),
    )

outliersCapping('avg_glucose_level')
outliersCapping('bmi')

for col in data.columns:
   if data[col].dtype=='object':
       l_en=LabelEncoder()
       data[col] = l_en.fit_transform(data[col])



x, y = data.iloc[:, 0:-1], data.iloc[:, -1:]
print("Before Oversampling, the counts of label 1: ", y.value_counts()[1])
print("Before Oversampling, the counts of label 0: ", y.value_counts()[0])
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)
print("After Oversampling, the counts of label 1: ", y.value_counts()[1])
print("After Oversampling, the counts of label 0: ", y.value_counts()[0])
data2=x.join(y)
data2

x = data2.drop(["stroke"], axis=1)
y = data2["stroke"].values
scaler = RobustScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)


finalmodel=XGBClassifier(max_depth= 80, max_features= 3, n_estimators= 50,learning_rate=0.1)
finalmodel.fit(x_train, y_train)
pickle.dump(finalmodel,open("model.pkl",'wb'))
y_predict = finalmodel.predict(x_test)
print("XGB")
print(finalmodel.score(x_train, y_train))
print(finalmodel.score(x_test, y_test))
print(classification_report(y_test, y_predict))

