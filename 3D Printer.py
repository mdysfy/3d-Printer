import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
data = pd.read_csv('data.csv')
data.info()



data.material = [0 if each == "abs" else 1 for each in data.material]
data.infill_pattern = [0 if each == "grid" else 1 for each in data.infill_pattern]
Y = data.material.values
X = data.drop(["material"],axis=1)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
#heatmap
fig, ax = plt.subplots(figsize = (20,7))
title = 'Correlation Heatmap of the 3D printing Dataset'
plt.title(title, fontsize = 20)
ttl = ax.title
sns.heatmap(data.corr(), cbar = True, cmap = 'cool', annot = True, linewidths = 1, ax = ax)
plt.show()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
score_list = []
for each in range(1,15):
    knn = KNeighborsClassifier(n_neighbors = each)
    knn.fit(x_train,y_train)
    score_list.append(knn.score(x_test,y_test))
    print("KNN {} nn score: {} ".format(each,knn.score(x_test,y_test)))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(x_train,y_train)
print("DecisionTree score: {}".format(dtree.score(x_test,y_test)))
from sklearn.ensemble import RandomForestClassifier
rfc_clf = RandomForestClassifier(n_estimators=15, random_state= 8)
rfc_clf.fit(x_train, y_train)
print("RandomForest score: {}".format(rfc_clf.score(x_test,y_test)))
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
ada_clf = AdaBoostClassifier(estimator=SVC(probability=True, kernel='rbf'),random_state=2)
ada_clf.fit(x_train, y_train)
print("adaboost score: {}".format(ada_clf.score(x_test,y_test)))
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
train_accuracy = logreg.score(x_train, y_train)
print("LogisticRegression Training accuracy:", train_accuracy)
test_accuracy = logreg.score(x_test, y_test)
print("Test accuracy:", test_accuracy)
test_preds_lr = logreg.predict(x_test)
print("Classification Report for Logistic Regression on Test Set:")
print(classification_report(y_test, test_preds_lr))
y_pred = logreg.predict(x_test)
print(y_pred)
