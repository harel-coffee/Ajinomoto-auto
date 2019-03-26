from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable

import math
import numpy as np

def plot_model_predictions(name,predicted,actual,log=False,ax=None):
    """Plots the predictions of a machine learning model.
    
    Create a scatter plot of machine learning model predictions vs.
    actual values from the data set along with a diagonal line showing
    where perfect agreement would be. 
    
    Args:
        name(str): The name of the value being predicted.
        
        predicted(array_like): The set of predicted values from a model
        
        actual(array_like): The set of actual values from the data set 
            which represent ground truth.
        
        log(bool,optional): If set to true the plot becomes a log-log 
            plot. Default False.
        
        ax(matplotlib.axes.Axes,optional): A preexisting axis object 
            where the plot will be located. Default None.
    
    Returns:
        matplotlib.axes.Axes: The axis object containing the created 
            scatterplot.
    
    """
    
    if log:
        predicted = [math.log(x) for x in predicted]
        actual = [math.log(y) for y in actual]

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    
    #Plot Scatter Plot
    ax.scatter(predicted,actual)
    ax.set_title(name + ' Predicted vs. Actual')

    #Plot Correct Ranges
    padding_y = (max(actual) - min(actual))*0.1
    min_y = min(actual)-padding_y
    max_y = max(actual)+padding_y
    ax.set_ylim(min_y,max_y)

    padding_x = (max(predicted) - min(predicted))*0.1
    min_x = min(predicted)-padding_x
    max_x = max(predicted)+padding_x
    ax.set_xlim(min_x,max_x)

    #Plot Diagonal Dotted Line
    dline = [min(min_y,min_x),max(max_y,max_x)]
    ax.plot(dline, dline, ls="--", c=".3")

    ax.set_xlabel('Predicted ' + name)
    ax.set_ylabel('Actual ' + name)
    
    return ax


def plot_model(model,data,targets,midpoint=0.1,title=None,zlabel=None,ax=None,pcs=None,plot_points=True):
    """Plots a heatmap representing a machine learning model and overlays training data on top.
    
    A heatmap of a machine learning model is generated to better understand how the model performs. 
    In order to deal with higher dimentional feature spaces, principal component analysis is used
    to reduce the feature space to the two dimensions with the most variance.  The data is then projected
    onto that plane and plotted over the model heatmap as a scatterplot.
    
    Args:
        model(sklearn.base.BaseEstimator): A scikit-learn style machine learning model.
    
        data(array_like): The feature space of the model training set. Dimensions are 
            (num_samples,feature_cardinality).
    
        targets(array_like): An array like object containing the ground truth for the model predictions.
            Dimensions are (num_samples,1).
    
        midpoint(float,optional): Select the midpoint value on the color map. Default 0.1.
    
        title(str,optional): Title of the generated plot. Default None.
    
        zlabel(str,optional): Label of the colormap axis. Default None.
    
        ax(matplotlib.axes.Axes,optional): Predefined axis used to draw the plot. Default None.
    
        pcs(array_like,optional): Specify Principal components to use for projection. Default None.
    
        plot_points(bool,optional): Overlay features as a scatterplot over the heatmap. Default True.
    
    """
    
    #Visualize Model
    #Create Principal Compoenents for Visualiztion of High Dimentional Space
    pca = PCA(n_components=2)
    if pcs is not None:
        pca.components_ = pcs
    
    data_transformed = pca.fit_transform(data)
    
    
    #Get Data Range
    xmin = np.amin(data_transformed[:,0])
    xmax = np.amax(data_transformed[:,0])
    ymin = np.amin(data_transformed[:,1])
    ymax = np.amax(data_transformed[:,1])

    #Scale Plot Range
    scaling_factor = 0.5
    xmin = xmin - (xmax - xmin)*scaling_factor/2
    xmax = xmax + (xmax - xmin)*scaling_factor/2
    ymin = ymin - (ymax - ymin)*scaling_factor/2
    ymax = ymax + (ymax - ymin)*scaling_factor/2

    #Generate Points in transformed Space
    points = 1000
    x = np.linspace(xmin,xmax,num=points)
    y = np.linspace(ymin,ymax,num=points)
    xv, yv = np.meshgrid(x,y)

    #reshape data for inverse transform
    xyt = np.concatenate((xv.reshape([xv.size,1]),yv.reshape([yv.size,1])),axis=1)
    xy = pca.inverse_transform(xyt)
    
    #Make Sure No Values are below 0
    zero_truncate = np.vectorize(lambda x: max(0.01,x))
    xy = zero_truncate(xy)
    
    #predict z values for plot
    z = model.predict(xy).reshape([points,points])
    minpoint = min([min(p) for p in z])
    maxpoint = max([max(p) for p in z])
    
    #Plot Contour from Model
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    
    scaled_targets = [target/max(targets)*200 for target in targets]
    
    #Overlay Scatter Plot With Training Data
    if plot_points:
        ax.scatter(data_transformed[:,0],
                    [1*value for value in data_transformed[:,1]],
                    c='k',
                    cmap=plt.cm.bwr,
                    marker='+',
                    s=scaled_targets,
                    linewidths=1.5
                    )
    
    ax.grid(b=False)

    midpercent = (midpoint-minpoint)/(maxpoint-minpoint)
    centered_cmap = shiftedColorMap(plt.cm.bwr, midpoint=midpercent)
    cmap = centered_cmap
    
    if midpercent > 1:
        midpercent = 1
        cmap = plt.cm.Blues_r
    elif midpercent < 0:
        midpercent = 0
        cmap = plt.cm.Reds
    
    z = [row for row in reversed(z)]
    im = ax.imshow(z,extent=[xmin,xmax,ymin,ymax],cmap=cmap)
    ax.set_aspect('auto')
    
    if title is not None:
        ax.set_title(title)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if zlabel is not None:
        plt.colorbar(im, cax=cax,label=zlabel)
    else:
        plt.colorbar(im, cax=cax)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap