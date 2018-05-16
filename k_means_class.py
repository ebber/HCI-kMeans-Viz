#!/Users/erikbeitel/Documents/EPFL/HCI/project/kMeans/bin/python

"""
Credit for alg goes to http://johnloeber.com/docs/kmeans.html

"""
import random
from base64 import b64decode
from json import loads
import numpy as np


#because osx is weird, see stackoverflow in case of more weirdness
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#set up mnist data
from mnist import MNIST
mndata = MNIST('./data')
images, labels = mndata.load_training()


from scipy import linalg as la
from graph_viz import Graph_Viz

class k_means:

    min_diff = 0.01

    def __init__(self,G, data):
        self.G = G
        #self.images = images
        self.labels = labels
        self.data = data
        for data_point in data:
            G.add_point(data_point[1], data_point[0])
            pass

    #do the thing
    def run(self, k):
        clusters, centroids = self.cluster(self.data, k, self.G)
        labelled_centroids = self.assign_labels_to_centroids(clusters, centroids)
        plt.close('all') #at this point we should have no figures
        return labelled_centroids


    def show_digit(self,im, label=None):
        """
        :param im: 784x1 vector thats a mnist data point
        :param label: what it is
        :return: nothing
        Displays the digit
        """
        fig = plt.figure()
        fig = plt.imshow(im.reshape(28,28))
        fig.set_cmap('gray_r')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        if label is not None:
            plt.title("label: " + str(label))
        plt.show(fig)
        plt.close(fig)



    #helper functions for lloyd's algorithm
    def init_centroids(self,labelled_data,k):
        """
        randomly pick some k centers from the data as starting values
        for centroids. Remove labels.
        """
        return map(lambda x: x[1], random.sample(labelled_data,k))

    def sum_cluster(self,labelled_cluster):
        """
        from http://stackoverflow.com/a/20642156
        element-wise sums a list of arrays.
        """
        # assumes len(cluster) > 0
        sum_ = labelled_cluster[0][1].copy()
        for (label,vector) in labelled_cluster[1:]:
            sum_ += vector
        return sum_

    def mean_cluster(self, labelled_cluster):
        """
        compute the mean (i.e. centroid at the middle)
        of a list of vectors (a cluster):
        take the sum and then divide by the size of the cluster.
        """
        sum_of_points = self.sum_cluster(labelled_cluster)
        mean_of_points = sum_of_points * (1.0 / len(labelled_cluster))
        return mean_of_points



    #2 main parts of llyods algorithm
    #forming clusters
    def form_clusters(self,labelled_data, unlabelled_centroids):
        """
        given some data and centroids for the data, allocate each
        datapoint to its closest centroid. This forms clusters.
        """
        # enumerate because centroids are arrays which are unhashable
        centroids_indices = range(len(unlabelled_centroids))

        # initialize an empty list for each centroid. The list will
        # contain all the datapoints that are closer to that centroid
        # than to any other. That list is the cluster of that centroid.
        clusters = {c: [] for c in centroids_indices}

        for (label,Xi) in labelled_data:
            # for each datapoint, pick the closest centroid.
            smallest_distance = float("inf")
            for cj_index in centroids_indices:
                cj = unlabelled_centroids[cj_index]
                distance = np.linalg.norm(Xi - cj)
                if distance < smallest_distance:
                    closest_centroid_index = cj_index
                    smallest_distance = distance
            # allocate that datapoint to the cluster of that centroid.
            clusters[closest_centroid_index].append((label,Xi))
        return clusters.values()

    #moving centroids
    def move_centroids(self, labelled_clusters):
        """
        returns list of mean centroids corresponding to clusters.
        """
        new_centroids = []
        for cluster in labelled_clusters:
            new_centroids.append(self.mean_cluster(cluster))
        return new_centroids

    #put it all in a repeating loop
    def repeat_until_convergence(self, labelled_data, labelled_clusters, unlabelled_centroids, viz):
        """
        form clusters around centroids, then keep moving the centroids
        until the moves are no longer significant.
        """
        previous_max_difference = 0
        num_reps=0
        while True:
            unlabelled_old_centroids = unlabelled_centroids



            unlabelled_centroids = self.move_centroids(labelled_clusters)
            labelled_clusters = self.form_clusters(labelled_data, unlabelled_centroids)
            labelled_centroids = self.assign_labels_to_centroids(labelled_clusters, unlabelled_centroids)
            sorted_labelled_centroids = labelled_centroids.sort(key= lambda cent:cent[0])
            #add to graph viz
            i=0
            #in case they are the same
            for cent in labelled_centroids:
                viz.add_centroid(cent[1], idx=i)
                i+=1

            # keep old_clusters and clusters so we can get the maximum difference
            # between centroid positions every time.
            differences = map(lambda a, b: np.linalg.norm(a-b),unlabelled_old_centroids,unlabelled_centroids)
            max_difference = max(differences)
            difference_change = abs((max_difference-previous_max_difference)/np.mean([previous_max_difference,max_difference])) * 100
            previous_max_difference = max_difference
            # difference change is nan once the list of differences is all zeroes.
            print(num_reps, difference_change)
            num_reps = num_reps+1
            if difference_change < self.min_diff or np.isnan(difference_change):
                break
        return labelled_clusters, unlabelled_centroids


    #label centroids
    def assign_labels_to_centroids(self, clusters, centroids):
        """
        Assigns a digit label to each centroid. Note: This function
         depends on clusters and centroids being in the same order.
        """
        labelled_centroids = []
        for i in range(len(clusters)):
            labels = map(lambda x: x[0], clusters[i])
            # pick the most common label
            most_common = max(set(labels), key=labels.count)
            centroid = (most_common, centroids[i])
            labelled_centroids.append(centroid)
        return labelled_centroids

    #wrap it
    def cluster(self, labelled_data, k, viz):
        """
        runs k-means clustering on the data.
        """
        centroids = self.init_centroids(labelled_data, k)
        clusters = self.form_clusters(labelled_data, centroids)
        final_clusters, final_centroids = self.repeat_until_convergence(labelled_data, clusters, centroids, viz=viz)
        return final_clusters, final_centroids









