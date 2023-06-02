# Implementation-of-SVM-For-Spam-Mail-Detection

# AIM

To write a program to implement the SVM For Spam Mail Detection.

# Equipments Required:

Hardware – PCs

Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.
5.End the program.


# Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Neha.MA
RegisterNumber:  212220040100
*/

import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
      result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()
data.info()
data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output

# chardet

![image](https://github.com/neha074/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113016903/8b7a70cf-9674-491a-a55f-bdade4df7aa7)

# head

![image](https://github.com/neha074/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113016903/bba7e9c3-3376-40ed-b8c6-cc0dfacb293f)

# info

![image](https://github.com/neha074/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113016903/71652aff-9054-4ea6-a5d4-ddd7f332d62d)

# is.null().sum()

![image](https://github.com/neha074/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113016903/04aa7241-b4af-4a51-9f0b-0e37a2eaefca)

# y_pred

![image](https://github.com/neha074/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113016903/ffe05d2d-e2da-40b5-b249-83ed8ac96647)

# accuracy

![image](https://github.com/neha074/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113016903/d73f1848-84e2-4138-93dd-ca21d40170f2)


## Result

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.















