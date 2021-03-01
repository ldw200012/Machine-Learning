import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("student-por.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "traveltime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train = None
x_test = None
y_train = None
y_test = None
best = 0
for i in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.02)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
            print(acc)

print(best)
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

style.use("ggplot")
x = "failures"
plt.scatter(data[x], data["G3"])
plt.xlabel(x)
plt.ylabel("Final Grade")
plt.title("Student Grades")
plt.show()

