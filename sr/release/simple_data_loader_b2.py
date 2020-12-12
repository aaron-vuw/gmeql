import tensorflow as tf
import numpy as np

class SimpleDataLoaderB2:
    def __init__(self, capacity=500):
        # Define the size of the dataset.
        self.capacity = capacity

        # Define all inputs and outputs variables for the dataset.
        self.x1 = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, 1], name="x1")
        self.x2 = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, 1], name="x2")
        self.x3 = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, 1], name="x3")
        self.y = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, 1], name="y")

        # Create dataset.
        self.x1Data = np.random.uniform(0., 2.1, (self.capacity, 1))
        self.x2Data = np.random.uniform(0., 2.1, (self.capacity, 1))
        self.x3Data = np.random.uniform(0., 2.1, (self.capacity, 1))
        self.yData = 0.8*np.power(self.x3Data, 4) + 0.8*np.power(self.x1Data, 3) + 1.2*np.power(self.x2Data, 2) + 1.4*self.x3Data
        return

    def getInputVariables(self):
        return [self.x1, self.x2, self.x3]

    def getOutputVariables(self):
        return [self.y]

    def getTrainingDataCapacity(self):
        return self.capacity

    def getTrainingDataset(self):
        dataset = {
            "input": {
                self.x1: self.x1Data,
                self.x2: self.x2Data,
                self.x3: self.x3Data
            },
            "output": {
                self.y: self.yData
            },
            "size": self.capacity
        }
        return dataset

    def getXTrain(self):
        return np.concatenate((self.x1Data, self.x2Data, self.x3Data), axis=1)

    def getYTrain(self):
        return self.yData

    def getTestingDataCapacity(self):
        return self.capacity

    def getTestingDataset(self):
        return self.getTrainingDataset()
