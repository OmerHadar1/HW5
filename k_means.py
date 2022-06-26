import numpy as np
from knn import KNN
import copy


class KMeans:
    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter
        self.groups = [[] for i in range(k)]
        self.centroids = None
        self.dict = {}

    def initialize(self, centroids):
        self.centroids = centroids
        for i in range(self.k):
            self.dict[i] = copy.deepcopy(centroids[i])

    def wcss(self):
        """

        :return:
        """
        counter = 0
        for i in range(self.k):
            if self.groups[i]:
                counter += int(np.sum((KMeans.distance_vec(self.groups[i], self.centroids[i]))**2, axis=1))
        return counter

    @staticmethod
    def distance_vec(vec_1, vec_2):
        """
        the methode calculate the distance between two numpy arrays
        :param vec_1: numpy array
        :param vec_2: numpy array
        :return: array
        """
        return np.sqrt(np.sum((vec_2 - vec_1) ** 2, axis=1))

    @staticmethod
    def distance_arr(centroids, points):
        """
        the method make an array of all the distance's between the points and the centroids, the columns index represent
        the centroid index and the rows index represent the point index
        :param centroids: array
        :param points: array
        :return: array
        """
        distance_array = KMeans.distance_vec(points, centroids[0])
        row = distance_array.shape[0]
        distance_array = distance_array.reshape(row, 1)
        for i in range(1, np.size(centroids, axis=0)):
            new_arr = KMeans.distance_vec(points, centroids[i]).reshape(row, 1)
            distance_array = np.hstack((distance_array, new_arr))
        return distance_array

    def new_centroids(self, lst):
        """
        the methode average the vectors inside the vector and return new vector
        :param lst:array
        :return:array
        """
        if lst[0]:
            new_arr = np.mean(lst[0][0], axis=0)
        else:
            new_arr = self.dict[0]
        for i in range(1, len(lst)):
            if lst[i]:
                new_vec = np.mean(lst[i][0], axis=0)
                new_arr = np.vstack((new_arr, new_vec))
            else:
                new_arr = np.vstack((new_arr, self.dict[i]))
        return new_arr

    @staticmethod
    def stop_fit(old_centroids, new_centroids):
        """
        the methode check if the new centroids are the same as the old centroids
        :param old_centroids: array 1-dim
        :param new_centroids: array
        :return: bool
        """
        val = True
        for m in range(old_centroids.shape[0]):
            bool_arr = new_centroids[m] == old_centroids[m]
            if np.sum(bool_arr) != bool_arr.shape[0]:
                val = False
        return val

    def fit(self, X_train):
        """
        the methode calculate a new centroids by the kmean algorithem
        :param X_train: array
        :return: dict
        """
        centroids = self.centroids
        for j in range(self.max_iter):
            centroids_groups = [[] for i in range(self.k)]
            dist_arr = self.distance_arr(centroids, X_train)  #יוצר מערך של המרחק מכל מרכז לכל נקודות, שורה= נקודה, עמודה=מרכז.
            closest_centroid = np.argsort(dist_arr)  # מסדר לפי אינדקסים מערך, כך נמצא לכל נקודה מה המרכז הכי קרוב אליה

            for k in range(X_train.shape[0]):  # לולא שעוברת על כל נקודה על מנת למצוא את המרכז הקרוב לה
                centroid_index = closest_centroid[k, 0]  #מציאת המרכז הקרוב
                if centroids_groups[centroid_index]:
                    centroids_groups[centroid_index][0] = np.vstack((centroids_groups[centroid_index][0], X_train[k]))
                else:
                    centroids_groups[centroid_index].append(X_train[k])  #הוספת הנקודה לקבוצת הנקודות שהכי קרובות לאותו מרכז

            self.groups = centroids_groups

            centroids = self.new_centroids(centroids_groups)  #מתודה ההמבצעת ממוצע על הנקודות ומוציאה מרכז חדש
            if self.stop_fit(self.centroids, centroids):
                break
            KMeans.initialize(self, centroids)

        return self.dict

    def predict(self, X):
        """
        the methode recive new points and return by there index the centroid ID that is the closest to them
        :param X: array
        :return: array
        """
        index_list = []
        dist_arr = KMeans.distance_arr(self.centroids, X)
        min_arr = np.argsort(dist_arr)
        for k in range(np.size(X, axis=0)):
            index_list.append(int(min_arr[k:k+1, 0:1]))
        return np.array(index_list)

