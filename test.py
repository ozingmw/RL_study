import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

array = np.array([[0,1,0,1], [9,8,7,6], [2,5,5,5], [8,5,4,4], [0,2,2,3]])
print(np.max(array, axis=1) == array)