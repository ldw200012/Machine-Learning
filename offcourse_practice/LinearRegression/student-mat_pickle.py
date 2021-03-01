import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv("../KNN/student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "traveltime", "studytime", "failures", "absences", ]]
predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best = 0
for i in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.01)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        with open("studentmodel2.pickle", "wb") as f:
            pickle.dump(linear, f)
            best = acc

print(best)

with open("studentmodel2.pickle", "rb") as f:
    linear2 = pickle.load(f)

predictions = linear2.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

plt.scatter(data["G1"], data["G3"])
plt.xlabel("G1")
plt.ylabel("G3")
plt.title("Student Grades")
plt.show()