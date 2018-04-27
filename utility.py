import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, linalg

def clean_proteomics(row):
    tp_row = row['Targeted Proteomics']
    tp_row.loc[row['TIR'] == 0]=0
    return tp_row

def clean_data(df,CoV_threshold=0.1,detection_limit=None):
    '''Cleans data by removing Replicates '''
    df_group = df.groupby([('Metadata','Cycle'),('Metadata','Strain'),('Metadata','Batch'),('Metadata','IPTG')])
    
    def delete_over_threshold(df):
        #Calculate CoV
        tdf = df
        tdf.loc[:,df.columns.isin([('Metadata','Notes')])] = 0
        mean_df = tdf.mean()
        std_df = tdf.std()
        cov_df = std_df/mean_df

        #print(list(df.columns))

        df.loc[:,(cov_df > CoV_threshold)&(df.columns.get_level_values(0)!='Metadata')] = float('NaN')
        
        return df
    
    #For Measurements above the CoV threshold set them to NaN.
    df = df_group.apply(delete_over_threshold)
    
    return df


def make_predictions(df,model_df,X_cols):
    '''Predict Performance of Strains From Models'''
    X = df.loc[:,X_cols]
    
    predictions = np.zeros((len(X),len(model_df)))
    for i,model in enumerate(model_df['Model']):
        predictions[:,i] = model.predict(X)
    
    return predictions


def partial_correlation(X,Y,Z):
    '''Perform a partial correlation analysis on X and Y controlling for Z.
    
    Take in samples from X, Y, and Z variables, 
    return the result from a conditional independence 
    test between X and Y and the error residuals for 
    plotting e_X and e_Y.
    '''
    
    beta_x = linalg.lstsq(Z, X)[0]
    beta_y = linalg.lstsq(Z, Y)[0]

    e_x = X - Z.dot(beta_x)
    e_y = Y - Z.dot(beta_y)
            
    corr, p_val = stats.pearsonr(e_x, e_y)
    
    return corr, p_val, e_x, e_y