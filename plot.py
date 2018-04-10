import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

def plot_corr(df,title='',mask_insignificant=True):
    sns.set(style="white")
    plt.figure(figsize=(14,8))
    n = len(df.columns)
    correction_factor= (n**2 - n)/2
    #correction_factor= 1
    
    # Compute the correlation matrix
    corr = round(df.corr()*100)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    #Generate a Mask for coefficients that are statistically insignificant
    p_vals = []
    for sp1 in df.columns:
        p_val_row = []
        for sp2 in df.columns:
            p_val = pearsonr(df[sp1].values,df[sp2].values)
            #print(sp1,sp2,p_val)
            p_val_row.append(p_val[1])
        p_vals.append(p_val_row)
    
    p_vals = [[val > (0.05/correction_factor) for val in row] for row in p_vals]
    mask = mask | np.array(p_vals)
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(corr,annot=True,square=True,mask=mask,fmt='0.3g',cmap=cmap,linewidths=.5)

    plt.title(title)
    plt.show()


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
    
def plotModel(model,data,targets,midpoint=0.1,pcs=None,title=None,zlabel=None,ax=None):
    '''Plots a 2d projection of the model onto the principal components.
       The data is overlayed onto the model for visualization.
    '''
    
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
    im = ax.imshow(z,extent=[xmin,xmax,ymin,ymax],cmap=cmap,aspect='auto')
    
    if title is not None:
        ax.set_title(title)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    if zlabel is not None:
        plt.colorbar(im, cax=cax,label=zlabel)
    else:
        plt.colorbar(im, cax=cax)
