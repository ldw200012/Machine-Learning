import pandas as pd
import sklearn
from sklearn import preprocessing, linear_model
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("car.data")
# print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
print(cls[0])#unacc
print(cls[1288])#acc
print(cls[1289])#good
print(cls[1292])#vgood


x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)

print(knn.score(x_test, y_test))

names = ["acc", "good", "unacc", "vgood"]
predictions = knn.predict(x_test)
for i in range(len(predictions)):
    print(names[predictions[i]], x_test[i], names[y_test[i]])
