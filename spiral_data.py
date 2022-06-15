import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
import pandas as pd
# import matplotlib.pyplot as plt

class SpiralDataCreator:
    def __init__(self, no_samples, type):

        self.N = no_samples

        multiplier = 20

        if type == "fibonacci":
            multiplier = 2

        

        theta = np.sqrt(np.random.rand(self.N))*multiplier # np.linspace(0,2*pi,100) *2*15 ### **1/2; /2
        beta = np.sqrt(np.random.rand(self.N))*multiplier
        
        if type == "blob":
            data_a = np.array([theta * 2, beta * 2]).T
            x_a = data_a + np.random.randn(self.N,2)

            data_b = np.array([-theta * 2, -beta * 2]).T
            x_b = data_b + np.random.randn(self.N,2)

            data_c = np.array([theta * 2, -beta * 2]).T
            x_c = data_c + np.random.randn(self.N,2)

        if type == "strange":
            data_a = np.array([np.sin(theta / 10) * theta, np.cos(theta * 2) * theta]).T
            x_a = data_a + np.random.randn(self.N,2)

            data_b = np.array([-theta * 2, -beta * 2]).T
            x_b = data_b + np.random.randn(self.N,2)

            data_c = np.array([theta * 2, -beta * 2]).T
            x_c = data_c + np.random.randn(self.N,2)



        if type == "fibonacci":
            theta = theta * 1.618

        if type == "archimedean" or type == "fibonnaci":
            r_a = theta
            data_a = np.array([np.cos(theta/2)*r_a, np.sin(theta/2)*r_a]).T
            x_a = data_a + np.random.randn(self.N,2)

            r_b = -theta
            data_b = np.array([np.cos(theta/2)*r_b, np.sin(theta/2)*r_b]).T
            x_b = data_b + np.random.randn(self.N,2)

            r_c = theta
            data_c = np.array([np.cos(theta/2)*r_c, np.sin(theta/2)*r_c]).T
            x_c = data_c + np.random.randn(self.N,2)

        if type == "circle":
            r_a = theta
            data_a = np.array([np.cos(theta/2)*60, np.sin(theta/2)*60]).T
            x_a = data_a + np.random.randn(self.N,2)

            r_b = theta
            data_b = np.array([np.cos(theta/2)*40, np.sin(theta/2)*40]).T
            x_b = data_b + np.random.randn(self.N,2)

            r_c = theta
            data_c = np.array([np.cos(theta/2)*20, np.sin(theta/2)*20]).T
            x_c = data_c + np.random.randn(self.N,2)

        """r_c = -2*theta + 2*pi
        data_c = np.array([(theta/2)*r_c, np.sin(theta/2)*r_c]).T
        x_c = data_c + np.random.randn(self.N,2)"""


        #circle
        """r_a = theta
        data_a = np.array([np.cos(theta/2)*100, np.sin(theta/2)*100]).T
        x_a = data_a + np.random.randn(self.N,2)"""


        """r_a = 2*theta + pi
        data_a = np.array([(theta**2)*r_a, (theta / 2)*r_a]).T
        x_a = data_a + np.random.randn(self.N,2)

        r_b = -2*theta - pi
        data_b = np.array([(theta**2)*r_b, (theta/2)*r_b]).T
        x_b = data_b + np.random.randn(self.N,2)

        r_c = 3*theta
        data_c = np.array([(theta - 10)*r_c, (theta / 2)*r_c]).T
        x_c = data_c + np.random.randn(self.N,2)"""

        res_a = np.append(x_a, np.zeros((self.N,1)), axis=1)
        res_b = np.append(x_b, np.ones((self.N,1)), axis=1)
        res_c = np.append(x_c, np.full((self.N,1), 2), axis=1)

        res = np.append(res_a, res_b, axis=0)
        res = np.append(res, res_c, axis=0)
        np.random.shuffle(res)

        self.df = pd.DataFrame(res, columns=['x', 'y', 'label'])
        self.x_a = res_a
        self.x_b = res_b
        self.x_c = res_c

"""datacreator = SpiralDataCreator(1000, 'circle')
df = datacreator.df
x_a = datacreator.x_a
x_b = datacreator.x_b
x_c = datacreator.x_c
plt.scatter(x_a[:,0],x_a[:,1], s=4)
plt.scatter(x_b[:,0],x_b[:,1], s=4)
plt.scatter(x_c[:,0],x_c[:,1], s=4)
plt.show()"""