import numpy as np
from knn import KNN
#from k_means import KMeans
import timeit as tm

# KNN

t0 = tm.default_timer()
train = np.genfromtxt('mnist_train_min.csv', delimiter=',', skip_header=1, dtype=np.int_)
test = np.genfromtxt('mnist_test_min.csv', delimiter=',', skip_header=1, dtype=np.int_)
y_train = train[:, 0]
X_train = train[:, 1:]
y_test = test[:, 0]
X_test = test[:, 1:]

clf = KNN(k=10)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
acc = np.sum(prediction == y_test) / len(y_test)
print(acc)
t1 =tm.default_timer()
print(t1-t0)

# KMeans
#data = np.genfromtxt('iris.csv', delimiter=',', skip_header=1, dtype=np.float_)[:,1:4]
#c1 = np.mean(data[:50],axis=0)
#c2 = np.mean(data[50:100],axis=0)
#c3 = np.mean(data[100:],axis=0)
#rng = np.random.default_rng()
#rng.shuffle(data, axis=0) # shuffle data rows
#train = data[:100]
#test = data[100:]
#kmeans = KMeans(k=3, max_iter=100)
#kmeans.initialize(np.array([c1,c2,c3]))
#class_centers = kmeans.fit(train)
#print(kmeans.wcss())
#classification = kmeans.predict(test)
