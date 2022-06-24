import numpy as np
from knn import KNN


class KMeans:
    def __init__(self, k, max_iter):
        KNN.__init__(self, k)
        self.max_iter = max_iter


    def initialize(self, centroids):
        self.label = [i for i in range(np.size(centroids, axis=0))]
        self.centroids = centroids


    def wcss(self):
        """

        :return:
        """
        counter = 0
        for i in range(len(self.label)):
            counter += np.sum(KMeans.distance_vec(self.centroids[i:i+1], self.centroids_groups[i:i+1]))
        return counter

        pass

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
        distance_array = KMeans.distance_vec(points, centroids[0:1])
        row = distance_array.shape[0]
        distance_array = distance_array.reshape(row,1)
        for i in range(1, np.size(centroids, axis=0)):
            distance_array = np.hstack((distance_array, KMeans.distance_vec(points, centroids[i-1, i]).reshape(row,1)))
        return distance_array

    @staticmethod
    def new_centroids(arr):
        """
        the methode average the vectors inside the vector and return new vector
        :param arr:array
        :return:array
        """
        new_arr = np.mean(arr[0:1], axis=0)
        for i in range(1, np.size(arr, axis=0)):
            new_vec = np.mean(arr[i-1:i], axis=0)
            new_arr = np.hstack((new_arr, new_vec))
        return new_arr

    def fit(self, X_train):
        """

        :param X_train:
        :return:
        """
        centroids = self.centroids
        for j in range(self.max_iter):  #מספר האינטרציות
            centroids_groups = []
            for i in self.label:  #יוצר רשימה של רשימות ריקות בגודל כמות המרכזים
                centroids_groups.append([])

            dist_arr = KMeans.distance_arr(centroids, X_train)  #יוצר מערך של המרחק מכל מרכז לכל נקודות, שורה= נקודה, עמודה=מרכז.
            closest_centroid = np.argsort(dist_arr)  # מסדר לפי אינדקסים מערך, כך נמצא לכל נקודה מה המרכז הכי קרוב אליה

            for k in range(np.size(X_train, axis=0)-1):  # לולא שעוברת על כל נקודה על מנת למצוא את המרכז הקרוב לה
                centroid_index = int(closest_centroid[k:k+1,0:1])  #מציאת המרכז הקרוב
                centroids_groups[centroid_index].append(X_train[k:k+1])  #הוספת הנקודה לקבוצת הנקודות שהכי קרובות לאותו מרכז

            centroids_groups = np.array(centroids_groups)  #נהפוך את הרשימה למערך על מנת לבצע פעולות numpy
            centroids = KMeans.new_centroids(centroids_groups)  #מתודה ההמבצעת ממוצע על הנקודות ומוציאה מרכז חדש

        centroids_dict = {}
        for index in range(np.size(centroids, axis=0)):
            centroids_dict[index] = centroids[index:index+1]
        self.centroids = centroids
        self.centroids_groups = centroids_groups
        return centroids_dict


    def predict(self, X):
        """

        :param X:
        :return:
        """
        index_list =[]
        for test_vec in X:
            dist_arr = np.sqrt(np.sum((self.centroids - test_vec)**2, axis=1))
            min_arr = np.argsort(dist_arr)[:self.k]
            index_list.append(min_arr)
        return np.array(index_list)

