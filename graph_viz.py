import plotly
#from Image import Image
from plotly.graph_objs import Layout, Scatter3d
import random
import numpy as np


import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Graph_Viz:

    #constants
    data_opacity =0.3

    data = {}

     #centroid -> xyz ->step

    layout = Layout(title="PCA of MNST")
    def __init__(self, num_centroids=9, reduction_matrix = np.eye(784,784), title= "Graph of MNST dataset"):
        self.centroids = [[list() for x in range(3)] for y in range(num_centroids)]
        self.reduction_matrix = reduction_matrix



    #internal helper functions

    def reduce_dims(self, data, dims=3):
        # carry out the transformation on the data using eigenvectors
        """
        :param data: a single image vector that is 784x1
        :param t_mat: the transformation matrix that is 784x784 (all the eig vectors)
        :param dims: how many dims to reduce too
        :return: the data down to 3 dims
        """
        t_mat = self.reduction_matrix[:, :dims]
        return np.dot(t_mat.T, data.T).T


    def save_img(self, im_v, cent_num, step):
        """

        :param im_v:image vector to save
        :return:
        """

        plt.figure()
        fig = plt.imshow(im_v.reshape(28,28))
        fig.set_cmap('gray_r')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        file_name = "centroid"+str(cent_num)+"-"+str(step)+".png"
        plt.savefig("images/" + file_name)
        plt.close()

    def add_centroid(self, im_v, idx):
        """
        add centroids
        :param im_v: the 874x1 im vector of the centroid
        :param label: which centroid, as an int
        """
        cord = self.reduce_dims(im_v)
        for i in range(len(cord)):
            self.centroids[idx][i].append(cord[i])

        self.save_img(im_v,idx, len(self.centroids[idx][0])-1 ) #0 is the x cord


    def add_point(self, im_v, label):
        """
        :param im_v: the 874x1 im vector of the centroid
        :return: nothing
        """
        cord = self.reduce_dims(im_v)
        if label in self.data:
            for i in range(0, 3):
                self.data[label][i].append(cord[i])

        else:
            self.data[label] = [ [cord[0]], [cord[1]], [cord[2]] ]



    #acting as model functions
    def get_step_count(self):
        return  len(self.centroids[0][0])

    def get_num_centroids(self):
        return  len(self.centroids)

    def get_cords_at(self,step):
        """
        :param: step number
        :returns: data [(label, [(x,y,z)]), (label, [(cords)])...num labels ]
        :returns: centroids [(x,y,z), (cord)... # centroids)
        """
        #background data always the same
        bg_data = []
        for key in self.data.keys():
            #will get rid of it
            cord = self.data[key]
            bg_data.append((key, cord))

        cent_data=[]
        for cent in self.centroids:
            cent_data.append( (cent[0][step],
                               cent[1][step],
                               cent[2][step]))

        return (bg_data, cent_data)






