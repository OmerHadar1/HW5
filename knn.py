import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k



    def fit(self, x_train, y_train):
        """
        the method make a dictionary, each label is an index and each index have all his vectors in a list
        :param x_train: numpy array
        :param y_train: numpy array
        :return: None
        """
        self.x_train = x_train
        self.y_trian = y_train



    def predict(self, x_test):
        """

        :param x_test:
        :return:
        """
        index_lst = []
        for test_vec in x_test:
            dist_arr = np.sqrt(np.sum((self.x_train - test_vec)**2, axis=1))
            min_arr = dist_arr.argsort()[:self.k]
            y_test = [self.y_trian[j] for j in min_arr]
            index = 0
            for i in y_test:
                if y_test.count(i) > index:
                    index = i
                else:
                    continue
            index_lst.append(index)
        return np.array(index_lst)









