import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_replicate_error(df,columns,log=True,group_on='Tak',threshold=None):
    '''Plot Replicate Error for a dataframe.  The DataFrame needs to contain Metadata headers'''
    #print(columns)
    CoV = lambda x: np.std(x) / np.mean(x) * 100 if np.mean(x) > 0 else 0
    group = df.groupby([('Metadata',group_on)])#[columns]
    #display(group[columns].agg([CoV,np.mean]).stack(1))
    percents,means = np.transpose(group[columns].agg([CoV,np.mean]).stack(1).values)
            
    #Create Figures
    plt.figure(figsize=(20,6))
    
    plt.subplot(1,2,2)
    sns.distplot(percents,norm_hist=True)
    plt.title('Percent Error Distribution (Mean Error: {:.1f}%)'.format(np.mean(percents)))
    plt.ylabel('Relative Frequency')
    plt.xlabel('Replicate Percent Error')
    plt.tight_layout()
    
    plt.subplot(1,2,1)
    if log:
        log_means = np.log10(means)
        plt.scatter(log_means,percents)
    else:
        plt.scatter(means,percents)
    plt.tight_layout()