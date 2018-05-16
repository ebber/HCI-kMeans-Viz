import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies
from dash.dependencies import Input, Output
from k_means_class import k_means
import flask

import random
from plotly.graph_objs import Layout, Scatter3d
app = dash.Dash()

from mnist import MNIST
mndata = MNIST('./data')
images, labels = mndata.load_training()
import numpy as np

from scipy import linalg as la
from graph_viz import Graph_Viz


#constants
import os
import glob
cwd = os.getcwd()
static_image_route = '/images/'
image_directory = cwd+static_image_route
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]



#animation constants
num_milli_per_step =4000 #*60 *60
current_step = 0
step_size = 4
is_play =False


def empty_folder(path):
    folder = path
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

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



def cords_to_traces(bg_data, cent):


    traces = []
    #bg data (MNST vizualized)
    for tup in bg_data:
        lab = tup[0]
        cords = tup[1]

        #will get rid of it
        color_val = str(random.randint(0,255))

        trace = Scatter3d(
            x=cords[0],
            y=cords[1],
            z=cords[2],
            #text=[str(lab) for i in range(len(cords[0]))], #ideally make this better but
            mode="markers",
            marker=dict(
                size=3,
                opacity=0.12
            ),
            name = str(lab),
            surfacecolor='rgb(' + color_val +',' + color_val +',' + color_val +')'
        )
        traces.append(trace)

    num_cent = len(cent)
    frame=Scatter3d(
        x=[cent[i][0] for i in range(num_cent)],
        y=[cent[i][1] for i in range(num_cent)],
        z=[cent[i][2] for i in range(num_cent)],
        mode='markers',
        marker=dict(color='black', size=15, opacity=1)
    )
    traces.append(frame)
    return traces


def get_filename_for_centroid(cent_num,step):
    file_name = "centroid"+str(cent_num)+"-"+str(step)+".png"
    return "images/"+file_name


#if have time, do this
def InteractiveImage(id, im_v):
    RANGE = [0, 255]
    encoded_image = im_v
    return dcc.Graph(
        id=id,
        figure={
            'data': [],
            'layout': {
                'xaxis': {
                    'range': RANGE
                },
                'yaxis': {
                    'range': RANGE,
                    'scaleanchor': 'x',
                    'scaleratio': 1
                },
                'height': 600,
                'images': [{
                    'xref': 'x',
                    'yref': 'y',
                    'x': RANGE[0],
                    'y': RANGE[1],
                    'sizex': RANGE[1] - RANGE[0],
                    'sizey': RANGE[1] - RANGE[0],
                    'sizing': 'stretch',
                    'layer': 'below',
                    'source': 'data:image/png;base64,{}'.format(encoded_image)
                }],
                'dragmode': 'select'  # or 'lasso'
            }
        }
    )

empty_folder(image_directory)
ims, data = prep_data(images, labels)
t_mat = get_PCA_matrix(ims)

G = Graph_Viz(num_centroids=10,reduction_matrix=t_mat)
engine = k_means(G, data)
centroids = engine.run(10)


#turn this into real data
#for i in range( 0,10):
#    G.add_centroid(data[i][1],0)
#for i in range( 10,0,-1):
#    G.add_centroid(data[i+10][1],1)

step=0
step_num=G.get_step_count()
num_centroids = G.get_num_centroids()
bg_data, cent = G.get_cords_at(step)
traces = cords_to_traces(bg_data,cent)
    #playing with animation

app.layout = html.Div(children=[
    html.Div(id='playing_div', style={'display':'none'}, title=True),
    html.Div(id='pausing_div', style={'display':'none'}, title=True),
    dcc.Graph(
        id='graph',
        figure={ 'data': traces,
            'layout': {
                'title': 'KMeans training on MNIST Data Visualization'
            }
        }
    ),
    html.Button('play', id='play-button'),
    html.Button('pause', id='pause-button'),
    dcc.Interval(id='auto-stepper',
                interval=1*num_milli_per_step, # in milliseconds
                 n_intervals=0
    ),
    dcc.Slider(
        id = "steper",
        min=1,
        max=step_num,
        value=1
    ),
    html.Div(id="centroid_images",className="row",
             children =[html.Img(id="im"+str(i)) for i in range(num_centroids)]
             )
])

@app.callback(
    dash.dependencies.Output('centroid_images', 'children'),
    [dash.dependencies.Input('steper', 'value')])
def update_image(step):
    children = []
    #global current_step
    #current_step = step
    for i in range(num_centroids):
        name = get_filename_for_centroid(i,step)
        im = html.Div( className="column",
                       style= {'width': str(100/num_centroids)+'%', 'display': 'inline-block'},
                        children = [
                            html.Div("Centroid :"+str(i)),
                            html.Img(id="im"+str(i), src=name, style={'width':'100%'})
                        ]
                    )
        children.append(im)
    return children

@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('steper', 'value')])
def update_figure(step):
        #each frame made by the outer for k in range(n) and should have all the shit for all centriods
    bg_data, cent = G.get_cords_at(step)
    traces = cords_to_traces(bg_data,cent)

    return {
        'data': traces,
        'layout': {
            'title': 'Dash Data Visualization'
        }
    }

#button shit
@app.callback(
    dash.dependencies.Output('steper', 'value'),
    [dash.dependencies.Input('auto-stepper', 'n_intervals')])
def on_interval(n_intervals):
    global current_step,is_play
    if is_play is True:
        current_step= (current_step+step_size)%step_num
    return current_step

@app.callback(
    Output('playing_div','title'),
    [Input('play-button', 'title')]
)
def on_play_click(title_of_button):
    global is_play
    is_play =True
    return True

@app.callback(
    Output('pausing_div','title'),
    [Input('pause-button', 'n_clicks')] )
def on_pause_click(n_clicks):
    global is_play
    is_play = False
    return False


# FIX: don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        pass
        #raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
