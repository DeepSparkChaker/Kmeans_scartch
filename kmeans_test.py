import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import style
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_blobs

class K_Means_Speed():
    def __init__(self , N_cluster, data):
        self.N_cluster= N_cluster
        self.data = data
        
    def fit(self,data):
        # 1. Randomly choose clusters
        rng = np.random.RandomState(0)
        i = rng.permutation(data.shape[0])[:self.N_cluster]
        centers = self.data[i]

        while True:
            # 2a. Assign labels based on closest center
            labels = pairwise_distances_argmin(self.data, centers)

            # 2b. Find new centers from means of points
            new_centers = np.array([self.data[labels == i].mean(0) for i in range(self.N_cluster)])

            # 2c. Check for convergence
            if np.all(centers == new_centers):
                break
            centers = new_centers

        return centers, labels

    def predict(self,data):
        labels = pairwise_distances_argmin(self.data, centers)
        return labels
if __name__ == '__main__':
	X, y = make_blobs(n_samples=300, centers=3, n_features=2,random_state=0)
	start = time.time()
	K_M =K_Means_Speed(N_cluster=3,data=X)
	centers, labels = K_M.fit(X)
	end = time.time()
	print("time of excution {}".format(end - start))
	print('centroids are : {}'.format(centers))
	print('first point:{} and the prediction cluster  is :{}'.format(X[0],labels[0]))