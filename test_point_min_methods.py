from sklearn.neighbors import KDTree
import numpy as np
import time

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.random((1000, 40))  # 1000 points in 40 dimensions
    Y = np.random.random((1000, 40))
    tree = KDTree(X, leaf_size=40)

    start = time.time()
    dist, ind = tree.query(Y, k=1, dualtree=True)
    kd_time = time.time() - start
    print("Time to run for KD tree: {}".format(kd_time))

