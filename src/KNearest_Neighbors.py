import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = [np.sqrt(np.sum((x-x_train)**2)) for x_train in self.X_train]
        
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            most_common = Counter(k_nearest_labels).most_common()
            y_pred.append(most_common[0])
        return y_pred
     
if __name__ == "__main__":
    pass