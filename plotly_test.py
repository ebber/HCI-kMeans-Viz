
#set up mnist data
from mnist import MNIST
mndata = MNIST('./data')
images, labels = mndata.load_training()
import numpy as np

from scipy import linalg as la
from graph_viz import Graph_Viz

def prep_data(images, labels):
    """
    take the output from mnist and and put it in a nicer format
    :return format:
    """
    # convert to numpy
    labels = np.array(labels)
    images = np.array(images)

    #consider getting rid of gray zones as per stackoverflow advice

    # make tuples of labels/images
    return images, zip(labels, images)

def get_PCA_matrix(data):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    from: stackoverflow.com/questions/13224362
    """
    from sklearn.preprocessing import StandardScaler
    m, n = data.shape
    # mean center the data
    data_std = StandardScaler().fit_transform(data)
    # calculate the covariance matrix
    cov_mat = np.cov(data_std.T)#, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmmetric,
    # the performance gain is substantial
    eig_vals, eig_vecs = la.eigh(cov_mat)
    # sort eigenvecs in decreasing order
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:,idx]
    # sort eigenvals according to same index
    eig_vals = eig_vals[idx]
    return eig_vecs


def reduce_dims(data, t_mat, dims=3):
    # carry out the transformation on the data using eigenvectors
    """
    :param data: a single image vector that is 784x1
    :param t_mat: the transformation matrix that is 784x784 (all the eig vectors)
    :param dims: how many dims to reduce too
    :return: the data down to 3 dims
    """
    t_mat = t_mat[:, :dims]
    return np.dot(t_mat.T, data.T).T

ims, data = prep_data(images, labels)
t_mat = get_PCA_matrix(ims)

G = Graph_Viz(num_centroids=2,reduction_matrix=t_mat)
for data_point in data:
    G.add_point(data_point[1], data_point[0])
for i in range( 0,10):
    G.add_centroid(data[i][1],0)
for i in range( 10,0,-1):
    G.add_centroid(data[i+10][1],1)
G.show_plot()

