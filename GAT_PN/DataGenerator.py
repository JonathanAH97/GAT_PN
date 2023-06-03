import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

class DataGenerator(object):

    def __init__(self):
        pass

    def gen_instance(self, max_length, dimension, seed=0): # Generate random TSP instance
        if seed!=0: np.random.seed(seed)
        sequence = np.random.rand(max_length, dimension) # (max_length) cities with (dimension) coordinates in [0,1]
        pca = PCA(n_components=dimension) # center & rotate coordinates
        sequence = pca.fit_transform(sequence) 
        return sequence

    def train_batch(self, batch_size, max_length, dimension): # Generate random batch for training procedure
        input_batch = []
        for _ in range(batch_size):
            input_ = self.gen_instance(max_length, dimension) # Generate random TSP instance
            input_batch.append(input_) # Store batch
        return input_batch

    def test_batch(self, batch_size, max_length, dimension, seed=0, shuffle=False): # Generate random batch for testing procedure
        input_batch = []
        input_ = self.gen_instance(max_length, dimension, seed=seed) # Generate random TSP instance
        for _ in range(batch_size): 
            sequence = np.copy(input_)
            if shuffle==True: 
                np.random.shuffle(sequence) # Shuffle sequence
            input_batch.append(sequence) # Store batch
        return input_batch

