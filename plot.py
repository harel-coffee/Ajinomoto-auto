import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from scipy.stats import pearsonr
import numpy as np

#import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

def plot_corr(df,title='',mask_insignificant=True):
    '''Plots a Correlation Grid'''
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

        
def strain_heatmap(df):
    '''Plot a Heatmap of All Strains Along Side TIR & Production'''
    
    sns.set_style('whitegrid')
    
    #Create Matrix of all catagories for the heatmap and Normalize by Column with a maximum at 1 and a min at 0
    #columns = list(df.columns[df.columns.get_level_values(0).isin(['TIR','Targeted Proteomics'])]) + [('GC-MS', 'dodecan-1-ol')]
    columns = [('TIR','sp|P69451|LCFA_ECOLI'),('Targeted Proteomics','sp|P69451|LCFA_ECOLI'),
               ('TIR','sp|Q41635|FATB_UMBCA'),('Targeted Proteomics','sp|Q41635|FATB_UMBCA'),
               ('TIR','tr|A1U2T0|A1U2T0_MARHV'),('Targeted Proteomics','tr|A1U2T0|A1U2T0_MARHV'),
               ('TIR','tr|A1U3L3|A1U3L3_MARHV'),('Targeted Proteomics','tr|A1U3L3|A1U3L3_MARHV'),
               ('TIR','sp|Q6F7B8|ACR1_ACIAD'),('Targeted Proteomics','sp|Q6F7B8|ACR1_ACIAD'),
               ('TIR','sp|P27250|AHR_ECOLI'),('Targeted Proteomics','sp|P27250|AHR_ECOLI'),
               ('GC-MS', 'dodecan-1-ol')
              ]
    
    col_norm = lambda col: col/max(col)
    #Group up Strains by Metadata (Average Across all Batches)
    heatmap_df = df.groupby([('Metadata','Cycle'),('Metadata','Strain'),('Metadata','IPTG')]).mean()
    
    #Select Only Rows With TIR, Targeted Proteomics, and Dodecanol data
    #display(heatmap_df[columns])
    heatmap_df = heatmap_df[columns].dropna()
    
    #Normalize and Sort By Production
    heatmap_df = heatmap_df.apply(col_norm,axis=0).sort_values(('GC-MS', 'dodecan-1-ol'))
    
    #Convert Zeros to NaNs
    heatmap_df = heatmap_df.replace(0, float('NaN'))

    
    plt.figure(figsize=(20,6))
    hm = sns.heatmap(heatmap_df.transpose(),cmap="viridis",cbar_kws={'ticks':[0,1]})
    plt.title('Strain Overview')
    plt.ylabel('')
    plt.xlabel('Strain')
    
    y_ticks = ['TIR: LCFA_ECOLI','Protein: LCFA_ECOLI',
               'TIR: FATB_UMBCA','Targeted Protein: FATB_UMBCA',
               'TIR: A1U2T0_MARHV','Protein: A1U2T0_MARHV',
               'TIR: A1U3L3_MARHV','Protein: A1U3L3_MARHV',
               'TIR: ACR1_ACIAD','Protein: ACR1_ACIAD',
               'TIR: AHR_ECOLI','Protein: AHR_ECOLI',
               'dodecanol'
              ]    
    y_ticks.reverse()
    
    cycles = heatmap_df.reset_index()[('Metadata','Cycle')]
    strains = heatmap_df.reset_index()[('Metadata','Strain')]
    x_ticks = ['{}-{}'.format(int(cycle),int(strain)) for cycle,strain in zip(cycles,strains)]
    
    ax = plt.gca()
    plt.xticks(rotation=45)
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    
    plt.tight_layout()
    plt.savefig('figures/strain_heatmap.png',dpi=600)
    
    sns.reset_defaults()
    

def quality_plot(df,assay_types,output_file=''):
    '''Visually Display Assay Quality Plots'''
    df_group = df.groupby([('Metadata','Cycle'),('Metadata','Strain'),('Metadata','Batch'),('Metadata','IPTG')])
    mean_df = df_group.mean()
    std_df = df_group.std()
    CoV_df = std_df/mean_df*100
    
    
    
    for assay in assay_types:
        means = np.log10(mean_df[assay]).values.flatten()
        CoVs = CoV_df[assay].values.flatten()
        
        finite_entries = np.logical_and(np.isfinite(means),np.isfinite(CoVs))
        
        means = means[finite_entries]
        CoVs =  CoVs[finite_entries]
        
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,2)
        sns.distplot(CoVs,norm_hist=True)
        plt.title('Percent Error Distribution (Mean Coefficient of Variation: {:.1f}%)'.format(np.mean(CoVs)))
        plt.ylabel('Relative Frequency')
        plt.xlabel('Replicate Coefficient of Variation')
        
        plt.subplot(1,2,1)
        plt.scatter(means,CoVs)
        plt.gca().set_axisbelow(True)
        plt.grid()

        plt.title('{} Replicate Error'.format(assay))
        plt.xlabel('Log10 Mean Measurement Value')
        plt.ylabel('Coefficient of Variation')
        
        plt.tight_layout()
        assay_nosp = assay.replace(' ','_')
        plt.savefig('figures/{}_error.png'.format(assay_nosp,dpi=600))
        plt.show()
        




def strain_plot3d(df,value=('GC-MS','dodecan-1-ol'),pathway=1,targets=None):
    if pathway == 1:
        #Pathway Name
        path_name='MAQU2220'
        X_target = np.transpose([[79841.880342,455256.1,373182.853333],
                    [21369.444444,309961.6,373182.853333]])
        scale = 100
        
    else:
        #Pathway Name
        path_name='MAQU2507'
        X_target = np.transpose([[151766.036,843892.720,698696.280],
                    [151766.036,620017.048,698696.280],
                    [151766.036,460305.120,579659.136]])
        scale = 12
        
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    #Define Pathway Protein Columns
    PATHWAY = df[('Metadata','Pathway')]==pathway
    proteins = df['Targeted Proteomics'].loc[:,df['TIR'].loc[PATHWAY].all(axis=0) > 0].columns
    
    #Define Selectors
    CYCLE_1 = df[('Metadata','Batch')] < 4
    CYCLE_2 = df[('Metadata','Batch')] > 3
    ZERO_VALUE = df[value]==0
    
    conditions = [(PATHWAY & CYCLE_1, 'black'),
                  (PATHWAY & CYCLE_2, 'red')]
    
    data = []
    for i,(condition,color) in enumerate(conditions):
        #Plot Non Zero Production Data Points
        X,Y,Z = [df.loc[condition & ~ZERO_VALUE]['Targeted Proteomics'][protein] for protein in proteins]
        label_fcn = lambda row: 'Strain {} Produced {:0.0f} mg/L Dodecanol'.format(int(row[('Metadata','Strain')]),row[('GC-MS','dodecan-1-ol')]*1000)
        label = df.loc[condition & ~ZERO_VALUE].apply(label_fcn,axis=1)
        S = df.loc[condition & ~ZERO_VALUE][value]*scale
        data.append(go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            mode='markers',
            name='Cycle {} Stains with Measureable Production'.format(i+1),
            text=label,
            marker=dict(
                size=S,
                symbol="x",
                color=color,
            
            )))
        
        #Plot Data Points with Zero Production
        if len(df.loc[condition & ZERO_VALUE]) > 0:
            label = df.loc[condition & ZERO_VALUE].apply(label_fcn,axis=1)
            X,Y,Z = [df.loc[condition & ZERO_VALUE]['Targeted Proteomics'][protein] for protein in proteins]
            data.append(go.Scatter3d(
                x=X,
                y=Y,
                z=Z,
                mode='markers',
                text=label,
                name='Cycle {} Stains with Zero Production'.format(i+1),
                marker=dict(
                    size=5,
                    color=color
                )))
    
    #Plot Target Data
    X,Y,Z = X_target
    data.append(go.Scatter3d(
                x=X,
                y=Y,
                z=Z,
                mode='markers',
                name='Proteomic Target for Cycle 2',
                marker=dict(
                    size=5,
                    symbol="circle-open",
                    color='blue'
                )))
    
    layout = go.Layout(
        scene = dict(
            xaxis=dict(title='x: {} [counts]'.format(proteins[0])),
            yaxis=dict(title='y: {} [counts]'.format(proteins[1])),
            zaxis=dict(title='z: {} [counts]'.format(proteins[2]))
            ),
        title='Proteomics vs. {} for {} strains'.format(value[1],path_name),
        #margin=dict(
        #        l=0,
        #        r=0,
        #        b=0,
        #        t=0),
        )
    
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='figures/Pathway{}StrainSummaryScatterPlot.html'.format(pathway))
    
    #Plot Proteomic Targets if Given
    #if targets is not None:
    #    pass
        
    #ax.scatter(pwc1df['LCFA_ECOLI'],pwc1df['FATB_UMBCA'],pwc1df[strain_proteins[i]],
    #           s=pathway_df['Dodecanol']*1000,marker='+',c='k')

    #Plot cycle1 data with zero production
    #temp_df = pwc1df.loc[pwc1df['Dodecanol'] == 0]
    #ax.scatter(temp_df['LCFA_ECOLI'],temp_df['FATB_UMBCA'],temp_df[strain_proteins[i]],
    #           s=50,marker='o',c='k')


    #Plot cycle2 data with nonzero production
    #temp_df = pathway_df.loc[pathway_df['Cycle']==2]
    #ax.scatter(temp_df['LCFA_ECOLI'],temp_df['FATB_UMBCA'],temp_df[strain_proteins[i]],
    #           s=temp_df['Dodecanol']*1000,marker='+',c='r')

    #Plot cycle2 data with zero production
    #temp_df = pathway_df.loc[(pathway_df['Cycle']==2)&(pathway_df['Dodecanol']==0)]
    #ax.scatter(temp_df['LCFA_ECOLI'],temp_df['FATB_UMBCA'],temp_df[strain_proteins[i]],
    #           s=50,marker='o',c='r')

    #Plot Targets
    #temp_df = target_dfs[i]
    #ax.scatter(temp_df['LCFA_ECOLI'],temp_df['FATB_UMBCA'],temp_df[strain_proteins[i]],
    #           s=50,marker='*',c='b')

    #Format Plot
    #plt.title('Proteomics vs. {}'.format(value[1]))
    #plt.xlabel(proteins[0])
    #plt.ylabel(proteins[1])
    #ax.set_zlabel(proteins[2])
    #plt.tight_layout()
    #plt.show()
    




#x2, y2, z2 = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
#trace2 = go.Scatter3d(
#    x=x2,
#    y=y2,
#    z=z2,
#    mode='markers',
#    marker=dict(
#        color='rgb(127, 127, 127)',
#        size=12,
#        symbol='circle',
#        line=dict(
#            color='rgb(204, 204, 204)',
#            width=1
#        ),
#        opacity=0.9
#    )
#)
#data = [trace1, trace2]
#layout = go.Layout(
#    margin=dict(
#        l=0,
#        r=0,
#        b=0,
#        t=0
#    )
#)
#fig = go.Figure(data=data, layout=layout)
#py.iplot(fig, filename='simple-3d-scatter')