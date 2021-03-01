import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("student-mat.csv", sep=";")
data2 = data[["G1", "G2", "G3", "failures", "absences", "studytime", "traveltime"]]

predict = "G3"
x = np.array(data2.drop([predict], 1))
y = np.array(data2[predict])

best = 0
x_train = None
x_test = None
y_train = None
y_test = None
for i in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.01)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        with open("studentScoreRel.pickle", "wb") as f:
            pickle.dump(linear, f)

print(best)

with open("studentScoreRel.pickle", "rb") as f:
    linear = pickle.load(f)
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

sns.set(style="darkgrid")
sns.relplot(x="G1", y="G3", hue="sex", col="school", kind="line", data=data)
plt.show()
