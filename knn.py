import numpy as np


class KNN:
    def __init__(self, k):
        """
        :param k: int type, the number of neighbors we want to check by
        """
        self.k = k

    def fit(self, x_train, y_train):
        """
        the method initialize group of vectors and a grop of labels for each vector, both groups have the same length,
        and each vector have its on label follow by there's index in the array
        :param x_train: numpy array
        :param y_train: numpy array
        :return: None
        """
        self.x_train = x_train
        self.y_train = y_train


    @staticmethod
    def most_frequent(lst):
        """
        the method find the most repeated object in the list
        :param lst: list
        :return: any class type that most frequent
        """
        counter = 0
        repeat_object = None
        for item in lst:
            if lst.count(item) > counter:
                counter = lst.count(item)
                repeat_object = item
        return repeat_object

    def predict(self, x_test):
        """
        the method recive new pionts and find to witch labael there belong to
        :param x_test: array
        :return:array
        """
        index_lst = []
        for test_vec in x_test:
            dist_arr = np.sqrt(np.sum((self.x_train - test_vec)**2, axis=1))
            min_arr = np.argsort(dist_arr)[:self.k]
            y_test = [self.y_train[j] for j in min_arr]
            index = KNN.most_frequent(y_test)
            index_lst.append(index)
        return np.array(index_lst)









