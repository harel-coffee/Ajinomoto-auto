#Base Imports
import os
import math
import itertools
import pandas as pd
import numpy as np

#Plotting Utilities
if __name__ == '__main__':
    #Allow Matplotlib to be used on commandline
    import matplotlib as mpl
    mpl.use('Agg')

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
plt.rcParams['font.size']=42

#Scipy Imports
from scipy.stats import norm,chi2,t
from scipy.optimize import differential_evolution

#Import Scikit learn functions
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, FunctionTransformer
from sklearn.feature_selection import RFECV,RFE
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

#Import Models
from sklearn.svm import SVR
from tpot import TPOTRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LassoLarsIC
from sklearn.neural_network import MLPRegressor

#For Rendering
from IPython.display import display,HTML
import pdfkit

#Handle Convergence Warnings
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from rectool.helper import *
from rectool.plot import *

#=========================================================#
# Reccomendation Class                                    #
#---------------------------------------------------------#
class recommendation_engine(object):
    """Recommends new strain designs given design specifications and a data set.
    
    Machine learning models are automatically fit strain data to an Electronic 
    Data Depot [EDD] formatted data file.  These models are then used to predict a 
    set of new strains which maximize the provided objective. Then, a report can 
    be generated and returned to the user.
    
    
    Args:
        data_file(str,optional): The path to an EDD style CSV file. This file contains 
            all the data on the initial set of strains. Default None.
       
        phase_space_file(str,optional): The path to a CSV file containing the allowable 
            designs. Defualt None.
    
        strain_file(str,optional): The path to a CSV containing the strain design metadata.
            Default None.
    
        features(list,optional): a list of the relavant features to use for model building
            present in the data_file. Default None.
    
        target(list,optional): The a list of the targets that the model is trying to 
            predict that are present in the data_file. Default None.
        
        objective_dict(dict,optional): A dictionary that defines the objective of the optimization.
            The dictionary must contain two keys, 'Objective' and 'Threshold'. The 'Objective' key
            can take values of {'maximize'|'minimize'|'target'}.  The 'Threshold' key takes a float
            value which determines the threshold for success. Default None.
    
        num_strains(int,optional): The number of strains to return to try in the next round.
             Default 3.
    
        engineering_accuracy(float,optional): How accurately engineers think they will be able
            to match the feature reccomendations. Default 0.
    
        engineering_base_strains(list,optional): A list of strains to base the predictions off of
            to minimize the difficulty of constructing new strains. Default None.
    
        seed(int,optional): A random seed to make the process deterministic. Default None.
    
        verbose(int,optional): A parameter that determines how much info to print during execution.
             0 nothing printed, 1 some printing of major events, 2 print everything. Default 0.
    
        tpot_dict(dict,optional): A dictionary that sets the TPOT parameters for picking the best
             model. Default {'generations':10,'population_size':50}.
    
        target_units(str,optional): The units of the target. Default 'g/L'.
    
        extrapolation_constant(float,optional): The scaling factor on the search area in the 
            feature space. This determines how far to extrapolate your optimization outside of
            the data set. Default 1.2.
    
    """
    
    intermediate_models = {}
    
    def __init__(self,
                 data_file = None,               # The EDD Data File Path 
                 phase_space_file = None,        # The Phase Space Data File Path
                 strain_file = None,             # The Stain Input file Relating Strains to phase space definitions
                 features=None,                  # List of Strings of the Data Features
                 target=None,                    # Target Feature String
                 objective_dict=None,            # Dictonary of the Objective
                 num_strains=3,                  # Number of Strain Predictions to Return
                 engineering_accuracy = 0,       # Accuarcy of being able to hit Proteomic Parameters
                 engineering_base_strains=None,  # Base Strains If Engineering to target Proteomics
                 seed = None,                    # A Random Seed to Make Art Deterministic for Debugging
                 verbose=0,                      # Display Information while Running? verbose=[0,1,2]
                 tpot_dict={'generations':10,'population_size':50}, #TPoT Settings as an optional argument
                 target_units = 'g/L',           # Units of the Target Metabolite
                 extrapolation_constant=1.2      # Multiple for extrapolation outside of data set
                ):
        
        #=================================================#
        # Check all Inputs and Initialize Class Variables #
        #-------------------------------------------------#

        #Load All data files And Add Variable Type to Main 
        #Data Frame (Independent,Dependent,Intermediate,Unused)
        if phase_space_file is not None:
            self.phase_space_df = pd.read_csv(phase_space_file)
            
            #Create Dictionary of all possible strain Parts
            self.strain_part_list = []
            for i,part in enumerate(self.phase_space_df['Type'].unique()):
                self.strain_part_list.append(self.phase_space_df['ID'].loc[self.phase_space_df['Type']==part].unique())
   
        else:
            self.phase_space_df = None
            
        self.df = None
        if strain_file is not None and data_file is not None:
            #Load Strain Designs
            self.df = pd.read_csv(strain_file)
            self.df = self.df.set_index('Line Name')
            self.df = pd.pivot_table(self.df,index=self.df.index,values='ID',columns='Part',aggfunc='first')
            
            #Set Columns of strain df as metadata
            self.df.columns = pd.MultiIndex.from_tuples([('Metadata',column) for column in self.df.columns])
            
            #Add Numerical Values for Independent Variables
            get_feature = lambda x: self.phase_space_df['Feature 1'].loc[self.phase_space_df['ID']==x].values[0]
            temp_df = self.df.applymap(get_feature)
            temp_df.columns = pd.MultiIndex.from_tuples([('Independent Variables',column[1]) for column in temp_df.columns])
            self.df = pd.merge(self.df,temp_df,how='left',left_index=True,right_index=True)
            
            #Load Strain Assay Data
            temp_df = pd.read_csv(data_file)
            temp_df = temp_df.set_index('Line Name')
            time_points = [col for col in temp_df.columns if is_number(col)]
            temp_df = pd.pivot_table(temp_df,index=temp_df.index,values=time_points[0],columns='Measurement Type')
            
            #Add headers to column
            new_cols = []
            for col in temp_df.columns:
                if features is not None and col in features:
                    new_cols.append(('Intermediate Variables',col))
                elif col in target:
                    new_cols.append(('Dependent Variable',col))
                else:
                    new_cols.append(('Unused Variables',col))
            temp_df.columns = pd.MultiIndex.from_tuples(new_cols)
            temp_df = temp_df.reindex(columns = ['Intermediate Variables','Dependent Variable','Unused Variables'], level=0)
            temp_df = temp_df.loc[:,temp_df.columns.get_level_values(0).isin(['Intermediate Variables','Dependent Variable'])]
            
            #Merge Assay Data into Dataframe
            self.df = pd.merge(self.df,temp_df,how='left',left_index=True,right_index=True)
            
        elif data_file is not None:
            #TODO: Implement Logic For when only a data file is present!
            self.df = pd.read_csv(data_file)
            self.df = self.df.set_index('Line Name')

            time_points = [col for col in self.df.columns if is_number(col)]
            self.df = pd.pivot_table(self.df,index=self.df.index,values=time_points[0],columns='Measurement Type')
        
            #Add headers to column
            new_cols = []
            for col in self.df.columns:
                if col in features:
                    new_cols.append(('Independent Variables',col))
                elif col in target:
                    new_cols.append(('Dependent Variable',col))
                else:
                    new_cols.append(('Unused Variables',col))
            self.df.columns = pd.MultiIndex.from_tuples(new_cols)
            self.df = self.df.reindex(columns = ['Independent Variables','Intermediate Variables','Dependent Variable','Unused Variables'], level=0)
            self.df = self.df.loc[:,self.df.columns.get_level_values(0).isin(['Independent Variables','Dependent Variable'])]
     
        #Load Objective Dict
        #TODO: Check that the right fields are in objective_dict
        if objective_dict is not None:
            self.objective = objective_dict['objective']
            self.threshold = objective_dict['threshold']
        
        self.engineering_accuracy=engineering_accuracy
        
        # Set Engineering Base Strains 
        # Note: Only Used if there is no phase space file
        self.engineering_base_strains = engineering_base_strains
        
        #Set the size of the cross validation set
        self.cv = min(10,len(self.df)-2)
        
        #Set Verbosity Level
        self.verbose = verbose
        
        self.features = features
        
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
        #Set TPOT Variables
        self.tpot_dict = tpot_dict
        
        #Set Target Units (Todo: Propagate these)
        self.target_units = target_units
        
        #Set Extrapolation Constant
        self.extrapolation_constant = extrapolation_constant
        
        self.initialize_models()
        
        #=================================================#
        # Check Inputs for Validity                       #
        #-------------------------------------------------#
        
        #TODO: Improve Error Checking here to report better errors!
        #Check To Make sure there is one and only one dependent variable
        assert(len(self.df['Dependent Variable'].columns)==1)
        
        #Check to make sure there is atleast one independent variable
        assert(len(self.df['Independent Variables'])>0)
        
        if self.verbose == 2:
            display(self.df)
        
        #=================================================#
        # Program Control Flow                            #
        #-------------------------------------------------#
        
        #If There is Data
        if self.df is not None:
            # Analyze Features and Get Correlations
            #TODO: Implement data analysis
            #self.analyze_data()
            
            # Fit Models
            self.fit_models()
            
            # Evaluate Models
            self.evaluate_models()
            
            #If there is a phase space file:
            if self.phase_space_df is not None:
                # Search for Best Strains In Phase Space
                self.optimal_strains(num_strains)
                
            #Else no phase space file
            else:
                # Search for Independent Variables which Maximize Production
                # Return Independent Variable Values
                # Return Engineering Strains which are closest to those values
                self.optimal_state(num_strains)
        
        #Else if there is no data and there is a phase space file
        elif self.phase_space_df is not None:
            #TODO: Generate A Set of Designs which uses parts evenly
            pass   
        
        #Otherwise nothing can be done!
        else:
            raise('Not Enough Information to do anything: Exiting Without Performing Action!')
    
    
    def initialize_models(self):
        """Initialize all of the models to use for prediction"""
        
        #==========================================================#
        # Define Default ML Models                                 #
        #----------------------------------------------------------#
            
        polynomialRegressor = Pipeline([('Scaler',StandardScaler()),
                                        ('Polynomial Features',PolynomialFeatures(degree=2, include_bias=True, interaction_only=True)),
                                        ('Feature Reduction',RFECV(Ridge(),cv=self.cv, scoring='r2')),
                                        ('Linear Regressor',BaggingRegressor(base_estimator=Ridge(),
                                                                             n_estimators=100, max_samples=.8,
                                                                             bootstrap=False,
                                                                             bootstrap_features=False,
                                                                             random_state=self.seed))])

        supportVectorRegressor = Pipeline([('Scaler',StandardScaler()),
                                           ('SVR',SVR())])

        LogScale = FunctionTransformer(np.log1p)
        LogRegressor = Pipeline([('Log Transform',LogScale),('Scaler',StandardScaler()),
                                 ('Linear',LassoLarsIC())])

        neural_model = Pipeline([('Scaling', StandardScaler()),
                                 #('FeatureSelection',RFECV(RandomForestClassifier())),
                                 #('FeatureSelection',BernoulliRBM()),
                                 ('NeuralNetwork', MLPRegressor(max_iter=10000,hidden_layer_sizes=(10,10,10),random_state=self.seed))])

    
        self.model_df = pd.DataFrame.from_dict({'Name':['Random Forest Regressor','Polynomial Regressor','Neural Regressor','TPOT Regressor'], 'Model':[RandomForestRegressor(),polynomialRegressor,neural_model,None]}).set_index('Name')
    
    
    def fit_models(self):
        """Fit Models to Provided Data
        
        A set of models are fit to the intermediate variables then the defined
        models are fit relating the intermediate and independent variables to the
        dependent variable.
        """
        
        # Find Intermediate Models
        X = self.df['Independent Variables'].values
        model = TPOTRegressor(generations=self.tpot_dict['generations'], population_size=self.tpot_dict['population_size'], cv=5, verbosity=self.verbose,random_state=self.seed)
        
        if 'Intermediate Variables' in self.df.columns.get_level_values(0):
            for var in self.df['Intermediate Variables'].columns:
                y = self.df[('Intermediate Variables',var)].values.ravel()
                self.intermediate_models[var] = model.fit(X,y).fitted_pipeline_
                self.intermediate_models[var].fit(X,y.ravel())
            
        # Find Composite Models
        X_comp = self.augment_state(X) #Build a composite state from the fit intermediate models
        y_comp = self.df['Dependent Variable'].values.ravel()
        self.model_df.at['TPOT Regressor', 'Model'] = model.fit(X_comp,y_comp).fitted_pipeline_
                
        #Fit Composite Models
        fit_model = lambda x: x.fit(X_comp,y_comp.ravel())
        self.model_df['Model'] = self.model_df['Model'].to_frame().applymap(fit_model)
        
        if self.verbose > 1:
            display(self.model_df)
    
        
    def augment_state(self,X):
        """Augment the state with intermediate model predictions.
        
        The feature vector is augmented by using it to make predictions with all the 
        intermediate models.  This expands the feature vector to include more information 
        in the data set. The augmented feature vector is returned.
        
        Args:
            X(array_like): an array of features to augment.
        
        Returns:
            array_like: an augmented feature vector containing predictions from intermediate
                models.
        """
        
        #Make sure state is in the right shape for prediction
        X_comp = np.array(X)
        #print(X_comp.shape,len(X_comp.shape))
        if 'Intermediate Variables' in self.df.columns.get_level_values(0):
            if len(X.shape)==1:
                X = np.reshape(X,(1,-1))
                X_comp = np.reshape(X_comp,(1,-1))
                
            for var in self.df['Intermediate Variables'].columns:
                y = np.reshape(self.intermediate_models[var].predict(X),(-1,1))
                X_comp = np.concatenate((X_comp,y),axis=1)
            #print(X)
            #print(X_comp)
        return X_comp
   
    
    def evaluate_models(self):
        """Add model statistics to self.model_df dataframe"""
        
        X = self.augment_state(self.df['Independent Variables'].values)
        y = self.df['Dependent Variable'].values.ravel()
        
        #Lambda functions to act on dataframe
        cross_validate = lambda x: cross_val_predict(x,X,y=y,cv=self.cv)
        mean_error = lambda y_predict: np.mean([y_a - y_p for y_a,y_p in zip(y,y_predict)])
        std_error = lambda y_predict: np.std([y_a - y_p for y_a,y_p in zip(y,y_predict)])
        mse = lambda y_predict: np.mean([(y_a - y_p)**2 for y_a,y_p in zip(y,y_predict)])
        
        
        self.model_df['Predictions'] = self.model_df['Model'].to_frame().applymap(cross_validate)
        self.model_df['Error Mean'] = self.model_df['Predictions'].to_frame().applymap(mean_error)
        self.model_df['Error Standard Deviation'] = self.model_df['Predictions'].to_frame().applymap(std_error)
        self.model_df['MSE'] = self.model_df['Predictions'].to_frame().applymap(mse)
        
        if self.verbose == 2:
            display(self.model_df)
            
    
    def optimal_strains(self,max_results=10):
        '''Find a set of optimal strains which maximize production'''
        
        #Find Best Model
        self.best_model = self.model_df['Model'].loc[self.model_df['MSE'].idxmin()]
        #display(self.best_model)
        
        #Find the Success Probability Function given the problem objective
        success_prob = lambda model: lambda p,x: self.success_prob(model,p,x,engineering_accuracy=self.engineering_accuracy)
        self.model_df['Target Interval'] = self.model_df['Model'].to_frame().applymap(self.calculate_target_interval)
        self.model_df['Success Probability'] = self.model_df['Model'].to_frame().applymap(success_prob)
        
            
        #display(self.model_df)
        
        self.strains = []
        self.probabilities = []
        self.production = []
        self.predictions = []
        for i in range(max_results):
            
            #Find bounds for Strains for differential evolution
            #Define Cost Given Currently Selected Points
            def cost(x):
                
                #Round off Strain Input
                strain = tuple([self.strain_part_list[i][int(part_num)] for i,part_num in enumerate(x)])
                
                #Check To make sure the strain is not the same as others in the set
                if strain in self.strains:
                    return 0
            
                #Convert Strain ID into Features
                features = self.augment_state(self.strains_to_features(strain))

                #Check to see success prob if not within dist threshold
                prob = 0
                n = len(self.model_df)
                deviations = self.model_df['MSE'].values
                models = self.model_df['Model'].values
                weights = calculate_weights(deviations)
                for model,weight,s_prob in zip(models,weights,self.model_df['Success Probability'].values):
                    p = model.predict(np.array(features).reshape(1, -1))[0]
                    prob += s_prob(p,x) * weight
                return -1*prob
        
            #Calculate Bounds for Strain Parts
            bounds = [(0,len(self.strain_part_list[i])) for i in range(len(self.strain_part_list))]
            sol = differential_evolution(cost,bounds,disp=self.verbose>0,seed=self.seed)
            
            strain = tuple([self.strain_part_list[i][int(part_num)] for i,part_num in enumerate(sol.x)])
            features = self.augment_state(self.strains_to_features([strain,]))
            self.strains.append(strain)
            self.probabilities.append(-1*sol.fun)
            self.predictions.append(features[0])
            self.production.append([model.predict(np.array(features).reshape(1, -1))[0] for model in self.model_df['Model']])
            
        # Put Results into Dataframe for Export calculate error and spread for each one... 
        self.strains = [list(strain) for strain in self.strains]
        columns = ['Part ' + str(key) for key in range(len(self.strain_part_list))] + ['Success Probability'] + [model for model in self.model_df.index]
        data = [feature + [prob,] + target for feature,prob,target in zip(self.strains,self.probabilities,self.production)]
        self.prediction_df = pd.DataFrame(data=data,columns=columns)
        #display(self.prediction_df)
    
    
    def optimal_state(self,max_results=10):
        '''Find a set of optimal features which maximize production'''

        #Define the State
        X = self.df['Independent Variables'].values
        
        #Find Best Model
        self.best_model = self.model_df['Model'].loc[self.model_df['MSE'].idxmin()]
        
        #Find the Success Probability Function given the problem objective
        success_prob = lambda model: lambda p,x: self.success_prob(model,p,x,self.engineering_accuracy)
        self.model_df['Target Interval'] = self.model_df['Model'].to_frame().applymap(self.calculate_target_interval)
        self.model_df['Success Probability'] = self.model_df['Model'].to_frame().applymap(success_prob)
        
            
        # Generate a set of predictions to meet objective
        bounds = create_bounds(X,padding_factor=self.extrapolation_constant)
        points = []
        probs = []
        production = []
        min_distance_between_points = self.distance_distribution(X)
        
        #Calculate Success Probability Functions for each model
        #target_interval = {}
        #success_prob = {}
        #for model in self.model_df['model']:
        #    target_interval[model] = self.calculate_target_interval(model)
        #    success_prob[model] = lambda p: self.success_prob(model,p,target_interval[model],engineering_accuracy)
            
        #print(target_interval)
        self.strains = []
        self.probabilities = []
        self.production = []
        self.predictions = []
        for i in range(max_results):
                
            #Define Cost Given Currently Selected Points
            def cost(x):
                    
                #Check to see if it is within distance threshold
                if len(self.predictions) > 0: 
                    if min(self.point_distance(x,self.predictions)) <= min_distance_between_points:
                        return 0
                        
                #Check to see success prob if not within dist threshold
                prob = 0
                n = len(self.model_df)
                deviations = self.model_df['MSE'].values
                weights = calculate_weights(deviations)
                for model,weight,success_prob in zip(self.model_df['Model'],weights,self.model_df['Success Probability']):
                    p = model.predict(np.array(x).reshape(1, -1))[0]
                    prob += success_prob(p,x) * weight
        
                return -1*prob
            
            sol = differential_evolution(cost,bounds,disp=self.verbose,seed=self.seed)
            self.predictions.append(sol.x.tolist())
            self.probabilities.append(-1*sol.fun)
            self.production.append([model.predict(sol.x.reshape(1, -1))[0] for model in self.model_df['Model']])
            
        # Put Results into Dataframe for Export calculate error and spread for each one... 
        columns = self.features + ['Success Probability'] + [str(model) for model in self.model_df.index]
        data = [feature + [prob,] + target for feature,prob,target in zip(self.predictions,self.probabilities,self.production)]
        self.prediction_df = pd.DataFrame(data=data,columns=columns)
            
        if self.verbose > 1:
            display(self.prediction_df)
        
    
    def generate_initial_strains(self,n):
        '''Create a list of n initial designs based on the provided phase space, return the list, and part usage statistics'''
        #Open Parts CSV from J5
        
        #Get List of Parts and enumerate statistics
        pass
        
    
    def __cross_validate(self,model):
        '''Calculate Error Residuals for Model For Reporting using 10-fold Cross Validation'''
        y_predict = cross_val_predict(self.models[model],self.X,y=self.y,cv=self.cv)
        score = cross_val_score(self.models[model],self.X,y=self.y,cv=self.cv)
        y_error = [y_a - y_p for y_a,y_p in zip(self.y,y_predict)]
        #y_error = [math.log(max(y_a,0.01)) - math.log(max(y_p,0.01)) for y_a,y_p in zip(self.y,y_predict)]
        #y_error = [[(y_p - y_a)/y_a for y_a,y_p in zip(self.y,y_predict)]]
        self.prediction_metrics[model] = {'predictions':y_predict,
                                          'residuals':y_error}
        self.residual_stdev[model] = np.std(y_error)
        self.residual_mean[model] = np.mean(y_error)
        
        
    def visualize_model_fits(self):
        '''Create Plots for a Set of Models'''
        models = self.model_df['Model']
        n = len(models)
        scale = 1.6
        fig = plt.figure(figsize=(15*scale,4*n*scale),dpi=300)
        X = self.augment_state(self.df['Independent Variables'].values)
        y = np.transpose(self.df['Dependent Variable'].values)[0]
        for i,model in enumerate(models):
            
            model_predictions = self.model_df['Predictions'].loc[self.model_df['Model']==model].values[0]
            model_residuals = [y_a - y_p for y_a,y_p in zip(y,model_predictions)]
            target_name = list(self.df['Dependent Variable'].columns)[0]
            model_name = list(self.model_df.loc[self.model_df['Model']==model].index)[0]
            #print(model_residuals)
            #print(y)
            #print(model_predictions)
            
            ax = fig.add_subplot(n,3,3 + i*3)
            sns.distplot(model_residuals,ax=ax)
            ax.set_title(model_name + ' Error Residuals')
            
            ax = fig.add_subplot(n,3,2 + i*3)
            
            plot_model_predictions(target_name,model_predictions,y,ax=ax)

            ax = fig.add_subplot(n,3,1 + i*3)
            plot_model(model,X,y,zlabel=target_name + ' (g/L)',title=model_name + ' fit',ax=ax)
        
        plt.tight_layout()
        return fig
        
    def __visualize_predictions(self,predictions,model,ax=None):
        '''Visualize Predictions In Comparison to The Data Set'''
        
        X = self.augment_state(self.df['Independent Variables'].values)
        y = np.transpose(self.df['Dependent Variable'].values)[0]
        #print(X)
        #print(y)
        target = self.df['Dependent Variable'].columns
        model_name = self.model_df.loc[self.model_df['Model']==model].index
        
        # Calculate Prediction Results
        #print(predictions)
        predicted_outputs = model.predict(predictions).tolist()
        all_states  =  predictions + X.tolist()
        all_outputs =  predicted_outputs + y.tolist()
        
        # Create Composite PCA with both Training Set & Predictions
        #from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_transformed = pca.fit_transform(all_states)
        
        # Plot the figure
        n = len(predictions)
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        
        plot_model(model,all_states,all_outputs,zlabel=target[0]+ ' (mg/L)',title=model_name + ' fit',plot_points=False,ax=ax)
        scaled_targets = [goal/max(all_outputs)*200 for goal in all_outputs]

        
        ax.scatter(data_transformed[0:n,0],
                    data_transformed[0:n,1],
                    c='b',
                    marker='+',
                    s=scaled_targets[0:n],
                    linewidths=1.5)
        
        ax.scatter(data_transformed[n:,0],
                    data_transformed[n:,1],
                    c='k',
                    marker='+',
                    s=scaled_targets[n:],
                    linewidths=1.5)
        
        #Add PCA Components explained variation to axis labels
        explained_variation = [str(round(var*100)) for var in pca.explained_variance_ratio_]
        ax.set_xlabel('Principal Component 1 (' + explained_variation[0] + '% of Variance Explained)')
        ax.set_ylabel('Principal Component 2 (' + explained_variation[1] + '% of Variance Explained)')
        
        #return fig

    
    def __visualize_success_probability(self,probabilities,ax=None):
        cprob = [0,]
        failure_prob = 100
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        
        for prob in probabilities:
            failure_prob *= (100 - prob)/100
            cprob.append(100-failure_prob)
        ax.plot(cprob)    
    
    
    def visualize_predicted_strains(self):
        scale = 1.6
        fig = plt.figure(figsize=(7.5*3*scale,5*scale),dpi=300)
        
        #Plot Success Probability
        ax = fig.add_subplot(1,2,1)
        self.__visualize_success_probability(self.probabilities,ax=ax)
        ax.set_title('Success Probability vs. Number of Strains Constructed')
        ax.set_xlabel('Number of Predicted Strains Constructed')
        ax.set_ylabel('Probability of at least one Strain Meeting Specification')
        
        
        #Plot Placement of new Predictions
        ax = fig.add_subplot(1,2,2)
        self.__visualize_predictions(self.predictions,self.best_model,ax=ax)
        ax.set_title('New Strain Predictions Overlayed on Best Model')
        
        return fig   

    
    #def fit(self,log_features=False):
    #    '''Fit Data to Specified Models using provided feature & target labels'''
    #    
    #    
    #    if log_features:
    #        log_mat = np.vectorize(math.log)
    #        self.X = log_mat(self.X)
 
    #    #self.y = self.edd_df.as_matrix(columns=target_indecies).transpose().tolist()[0]
    #    #Use Tpot to find models
    #    self.X = np.array(self.X)
    #    self.y = np.array(self.y)
    #    print(max(self.y),min(self.y))
    #    print(self.X.shape)
    #    print(self.y.shape)
    #    
    #    #Fit TPOT Model
    #    #self.models['TPOT'] = TPOTRegressor(generations=20, population_size=50, cv=5, verbosity=self.verbose)
    #    self.models['TPOT'] = TPOTRegressor(generations=2, population_size=5, cv=5, verbosity=self.verbose)
    #    self.models['TPOT'] = self.models['TPOT'].fit(self.X, self.y).fitted_pipeline_
    #    
    #    if self.verbose > 0:
    #        print(self.models['TPOT'].score(self.X, self.y))
    #    
    #    #Cross Validate & Fit Models
    #    for model in self.models:
    #        self.__cross_validate(model)
    #        self.models[model].fit(self.X,self.y)
    #    
    #    if self.verbose > 1:
    #        self.visualize_model_fits(self.models)
    #        plt.show()
    
    
    def success_prob(self,model,point,x,engineering_accuracy=0.10):
        '''Calculate the probability of hitting a target interval given a point in the feature space.
           Engineering error is incorporated into the calculation so that nearby points are taken into account.
        
        Gather Predictions in an evenly spaced ball about the target point based on engineering accuracy
        Sample until estimates of mean and standard deviation are known to 1% error in 95% confidence intervals
        Report mean & worst case standard deviation at 95% convidence intervals
        '''

        target_interval = self.model_df['Target Interval'].loc[self.model_df['Model']==model].values[0]
        residual_stdev = self.model_df['Error Standard Deviation'].loc[self.model_df['Model']==model]
        residual_mean = self.model_df['Error Mean'].loc[self.model_df['Model']==model]
        skl_model = self.model_df['Model'].loc[self.model_df['Model']==model].values[0]
        
        #Calculate the Moments of the Distribution for Engineering
        if engineering_accuracy == 0:
            mu = point
            sigma = 0
        else:
            #Change this path to support engineering accuracy values in new architecture
            mu,sigma = ball_target_moments(x,engineering_accuracy,skl_model,sample_points=100,seed=self.seed)
                            
        sig = math.sqrt(sigma**2 + residual_stdev**2)
        mu = mu + residual_mean
            
        #Calculate The Probability of hitting the target interval (Add the influence of the point P)
        prob = (norm.sf(target_interval[0], mu, sig) - norm.sf(target_interval[1], mu, sig))*100
        
        return prob

    
    def strains_to_features(self,strain_parts):
        '''Take a list of parts and return the feature vector associated with that strain'''
        #display(self.phase_space_df['Feature 1'])
        #print(strain_parts)
        #print(self.phase_space_df['Feature 1'].loc[self.phase_space_df['ID'] == strain_parts[0][0]])
        part_to_feature = lambda x: self.phase_space_df.loc[self.phase_space_df['ID']==x,'Feature 1'].values[0]
        #print(part_to_feature(strain_parts[0]))
        part_to_feature = np.vectorize(part_to_feature)
        #print(part_to_feature(strain_parts))
        return part_to_feature(strain_parts)

    
    def strains_to_target(self,strain_ids):
        #Get Time Points
        time_points = [col for col in self.data_df.columns if is_number(col)]
        #display(self.data_df)
        #print(time_points[0],type(time_points[0]))
        y = self.data_df.loc[self.data_df['Measurement Type']==self.target[0],time_points[0]].values
        return y
        
    
    def calculate_target_interval(self,model):
        '''Figure out what the goal interval is for declaring success given a model'''
        
        X = self.augment_state(self.df['Independent Variables'].values)
        y = self.df['Dependent Variable'].values
        if self.objective == 'maximize':
            #Set Success Condition and Cost for Interval Search
            cost = lambda x: -1*model.predict(np.array(x).reshape(1, -1))[0]
            success_val = max(y)*(1+self.threshold)
            
            #Find Value Which Minimizes Cost
            sol = differential_evolution(cost,create_bounds(X,padding_factor=1.2),disp=False,seed=self.seed)
            target_interval = [min(success_val,-1*sol.fun),math.inf]
        else:
            raise ValueError('Only Maximize Objective is Currently Implemented')     
        
        return target_interval    
 
    
    def summarize_model_statistics(self):
        best_model = self.model_df.loc[self.model_df['MSE'].idxmin()].name
        best_model_95 = self.model_df['MSE'].loc[self.model_df.index==best_model].values[0]*2
        #print(list(self.df['Dependent Variable'].columns)[0])
        #print(best_model_95)
        units = self.target_units
        #print('target: {}'.format(self.df['Dependent Variable'].columns[0])) 
    
        html  = '<p> The following models have been evaluated:'
        for model,row in self.model_df.iterrows():
            html += '{},'.format(model) 
            html += '. '
        html += 'For the best model ({}), 95% of the observations fall within {:.3f} ({}) of {} production.'.format(best_model,best_model_95,units,list(self.df['Dependent Variable'].columns)[0])

    
        for model,row in self.model_df.iterrows():
            #print(model,row)
            html += 'The {} residuals have a standard deviation of {:.3f} ({}) and the mean of the predictions is offset by {:.3f} ({}). '.format(model,row['MSE'],self.target_units,row['Error Mean'],self.target_units)
        html += '</p>'
    
        return html
    
    def generate_report(self, artifacts_dir=None, output_file=None):
        '''Generate a PDF report of the ART Output'''
        
        y = self.df['Dependent Variable'].values
        #print(y,np.max(y))
        target = list(self.df['Dependent Variable'].columns)[0]
        #First Generate HTML
        markdown = ''
                
        # Generate Header & Intro Paragraph Based on Objective
        if self.objective == 'minimize':
            markdown += '<h1>Minimize ' + target + ' Production</h1>\n'
                
        elif self.objective == 'maximize':
            markdown += '<h1>Predictions to Maximize ' + target + ' Production</h1>\n'
            markdown += '<p>In this data set the maximum value is {:.3f} ({}) of {}.'.format(np.max(y),self.target_units,target) 
            markdown += 'The objective is to predict strain designs which will maximize production.  '
            markdown += 'Successful design is to exceed the maximum observed production by {:.1f}% which is {:.3f} ({}).</p>\n'.format(self.threshold*100,(1 + self.threshold)*np.max(y),self.target_units)
        
        elif self.objective == 'target':
            markdown += '<h1>Target ' + str(self.production) + '(mg/L) ' + target + ' Production</h1>\n'
        
        #Figure 1. Strain Probabilities Visualization
        fig1 = self.visualize_predicted_strains()
        caption1 = summarize_success_probability(self)
        
        #print(markdown)
        #print(type(markdown))
        if output_file is None:
            display(HTML(markdown))
            plt.show(fig1)
            markdown = ''
            markdown += caption1
        else:
            file_path = os.path.join(artifacts_dir, 'figure1.png')
            markdown += create_figure_markdown(fig1, caption1, file_path)
                
        #Table 1. A List of all strains
        markdown += '<h2>Predictions</h2>'
        #markdown += '<p>First as a point of comparision, the best performing strain in the dataset is presented.</p>'
        #markdown += '<div class=\'table\'>' + self.edd_df.loc[self.edd_df[self.time_points[0],self.target[0]] == max(self.y)].to_html() + '</div>'
        markdown += '<p>The Table Below provides a set of predictions which if followed should maximize the chance of successful strain engineering.</p>'
        markdown += '<div class=\'table\'>' + self.prediction_df.to_html() + '</div>'
        
        #Table 2. (If there are engineering Base Strains Specified) Calculate Changes to base closest base strain needed...
        #TODO: If needed Remove Brute Force Approach to finding closest strain (Once strain numbers get large)
        if self.engineering_base_strains is not None:
            
            columns = ['Base Strain'] + self.features
            data = []
            #print(self.features)
            #print(list(self.df.columns))
            edd_df = self.df['Independent Variables']
            engineering_df = edd_df[self.features].loc[edd_df.index.isin(self.engineering_base_strains)]
            engineering_strains = engineering_df.values
            engineering_strain_names = engineering_df.index.values
            predicted_strains = self.prediction_df[self.features].values
            for target_strain in predicted_strains:
                
                #find closest engineering strain
                min_cost = math.inf
                min_strain = None
                min_strain_name = None
                for strain_name,base_strain in zip(engineering_strain_names,engineering_strains):
                    cost = lambda X,Y: sum([(x-y)**2 for x,y in zip(X,Y)])
                    #print(target_strain,base_strain,cost(target_strain,base_strain))
                    strain_cost = cost(target_strain,base_strain)
                    if strain_cost < min_cost:
                        #print('New Minimum!',strain_name)
                        min_strain = base_strain
                        min_strain_name = strain_name
                        min_cost = strain_cost
                    
                #Calculate Fold Changes Needed To Reach Target Strain
                def fold_change(base_strain, target_strain):
                    fold_changes = []
                    for base_val,target_val in zip(base_strain,target_strain):
                        pos_neg = 1
                        if base_val > target_val:
                            pos_neg = -1
                        
                        fold_changes.append(pos_neg*max(base_val,target_val)/min(base_val,target_val))
                    return fold_changes
                
                
                line = [min_strain_name] + fold_change(min_strain,target_strain)
                data.append(line)
            
            engineering_df = pd.DataFrame(data,columns=columns)
            
            markdown += '<h2>Engineering Guidelines</h2>'
            markdown += '<p>The Table Below provides instructions on how to use the base strains to most easily reach the new predictions. Fold changes are expressed for each protein with respect to the base strain listed.</p>'
            markdown += '<div class=\'table\'>' + engineering_df.to_html() + '</div>'
            
        
            #Add Data Frame to HTML for Printing
        
        #Figure 2. Model Fit Plots with Caption Statistitics 
        fig2 = self.visualize_model_fits()
        caption2 = self.summarize_model_statistics()
        markdown += '<h2>Model Evaluation</h2>'
        if output_file is None:
            display(HTML(markdown))
            plt.show(fig2)
            markdown = ''
            markdown += caption2
        else:
            file_path = os.path.join(artifacts_dir, 'figure2.png')
            markdown += create_figure_markdown(fig2,caption2, file_path)
        
        #Figure 3. Data Relationships (Future)        
        
        #Print to Jupyter Notebook if output file is none
        if output_file is None:
            display(HTML(markdown))
        else:
            #Figure out extension
            _ , extension = os.path.splitext(output_file)
            extension = extension[1:].lower()
            if extension == 'pdf':
                #Render PDF from HTML with CSS Report Stylesheet
                options = {
                        'dpi':96,
                        'page-size': 'Letter',
                        'margin-top': '0.75in',
                        'margin-right': '0.75in',
                        'margin-bottom': '0.75in',
                        'margin-left': '0.75in',
                        'encoding': "UTF-8",
                        'no-outline': None,
                        }

                art_package_dir = os.path.dirname(os.path.abspath(__file__))
                css_path = os.path.join(art_package_dir, 'report.css')
                pdfkit.from_string(markdown, output_file, options=options, css=css_path) #css=stylesheet

            elif extension == 'csv':
                self.prediction_df.to_csv(output_file)
                
            else:
                raise ValueError('Invalid File Extension. Cannot Save out Report in Format:' + str(extension))

    def point_distance(self,X,points):
        '''Calculate Distance between one point and a set of points'''
        return [dist(X,Y) for Y in points]
        
    
    def distance_distribution(self,points,display=False,threshold=0.6):
        '''Visualize the distances between all of the proteomic points to find a good threshold for guesses'''
        distances = sorted([dist(x,y) for x, y in itertools.combinations(points, 2)])
        if display:
            sns.distplot(distances)
            plt.show()
        min_dist_index = int(len(distances)*(1-threshold))
        return distances[min_dist_index]

  
#If we run from the commandline    
if __name__ == "__main__":
        
    import argparse
    parser = argparse.ArgumentParser()
    
    #Optional File Inputs
    parser.add_argument('-df','--data_file',type=str, help='EDD format CSV file path for strain data.')
    parser.add_argument('-pf','--phase_space_file',type=str,help='CSV file path containing strain design constraints.')
    parser.add_argument('-sf','--strain_file',type=str,help='CSV file path containing strain design metadata.')
    
    #Simulation Settings
    parser.add_argument('-f', '--features',nargs='+',type=str,help='A list of features to use in simulation.')
    parser.add_argument('-tg', '--target',type=str,help='The name of the target to be optimized.')
    parser.add_argument('-tu', '--target_units',type=str,help='Units of the target for reporting.',
        default='g/L')
    parser.add_argument('-n', '--num_strains',type=int,help='The number of strains to reccomend from simulation.',
        default=3)
    
    parser.add_argument('-ea','--engineering_accuracy',type=float,help='The precision which biologists believe \
        they can hit the prediction. (0.5 means hitting within 50%% of the target).',default=0)
    parser.add_argument('-ebs','--engineering_base_strains',type=str,nargs='+',help='Strain IDs which \
         predictions are based on. If not specified use any strain in the data set to modify.')
    
    parser.add_argument('-obj', '--objective',type=str,choices=['minimize','maximize','target'],
        help='The optimization objective for the target provided.',default='maximize')
    parser.add_argument('-th', '--threshold',type=float,
        help='The success criteria threshold. (eg. If the objective is to maximize the target, \
        a threshold of 1.2 means that the target must be improved by 20%% to qualify as a \
        success.',default=0.0)
        
    parser.add_argument('-ext','--extrapolation_constant',type=float,help='Allowable distance \
        multiplier to extend the optimizer search area outside of the given strains.  (i.e. \
        How much can we extrapolate from the data.) 1.0 means no extrapolation, 1.2 means \
        20%% extrapolation.',default=1.2)
    
    #TPOT Arguments
    parser.add_argument('-gen','--generations',type=int,help='Number of generations to run tpot \
        models for.',default=10)
    parser.add_argument('-pop','--population_size',type=int,help='Population size used for TPOT \
        genetic algorithm.',default=50)
    
    #General Options
    parser.add_argument('-s', '--seed',type=int,help='Set Random Seed.')
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
                        help="increase output verbosity")
    
    #Output Argument
    parser.add_argument('-of', '--output_file', type=str, 
        help='where to write output file. Possible extensions are: [*.csv, *.pdf]')
    
    args = parser.parse_args()
    
    #Errors to Throw: (Do this in class?)
    # If features provided, data file must exist?
    # If Features provided, Target Must also be provided.
    # Features must be in the data file
    # Target Must be in the data file
        
    #Create Art Params Dictionary to Feed reccomendation_engine class.
    art_params = {}
    
    #File Params
    art_params['data_file'] = args.data_file
    art_params['phase_space_file'] = args.phase_space_file
    art_params['strain_file'] = args.strain_file
    
    #Simulation Params
    art_params['features'] = args.features
    art_params['target']   = [args.target]
    art_params['objective_dict'] = {'objective':args.objective,'threshold':args.threshold}
    art_params['num_strains'] = args.num_strains
    art_params['engineering_accuracy'] = args.engineering_accuracy
    art_params['engineering_base_strains'] = args.engineering_base_strains #['B-Mm','B-Mh','B-Hl','B-Hm','B-Hh','BL-Ml','BL-Mh','B-Ll','B-Lm','B-Ml','BL-Mm']
    art_params['verbose'] = args.verbosity
    art_params['tpot_dict'] = {'generations':args.generations,'population_size':args.population_size}
    art_params['seed']=args.seed
    art_params['target_units'] = args.target_units
    art_params['extrapolation_constant'] = args.extrapolation_constant
        
    #Run Art
    art = recommendation_engine(**art_params)
    
    #Return Report
    art.generate_report(args.output_file)
   
