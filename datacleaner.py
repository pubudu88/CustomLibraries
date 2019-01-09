
'''
Input parameters

df_data : The data frame which includes independent variables and the target variable
remove_cols : Specify which columns you need to remove from the model. Example : customer name, date etc. This should be a list

missing_thres : The threshold of % of missing values in a given features which is condsidered before removing them

df_catOrNum : Prior to running this finction you need to create a dataframe which inlcude the data type of all the features. This data frame should have two columns (Feature, Feature_type) feature type can have 'Categorical' or 'Numerical'

standardisation_needed :  If you need to standardise features set this value to 1 or else to 0. If you use a non linear model like decision trees, you dont need to standardise features

save_data_scoring : If you need to save the cleaned data + all the data which is required at scoring stage, set this to 1 or else 0

target : specify the target variable name

model_name : Give a name to the model you are going to build

'''





import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings('ignore')

import sqlalchemy
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error

from sklearn.grid_search import GridSearchCV  


from sklearn import preprocessing
import datetime
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import kurtosis


def data_cleaning(df_data,remove_cols,missing_thres,df_catOrNum,standardisation_needed,save_data_scoring,target,model_name):
    

    for i in remove_cols:
        try:
            del df_data[i]
        except:
            continue
            
    
    def show_missing(data):
        missing = data.columns[data.isnull().any()].tolist()
        missing_series=(data[missing].isnull().sum())
        missing_df=pd.DataFrame({'col_name':missing_series.index, 'missing_count':missing_series.values})
        missing_df.sort_values(by='missing_count',ascending=False, inplace=True)
        return missing_df


    features_all_missing = df_data.loc[:,df_data.notnull().sum()==0].columns
    df_data.drop(features_all_missing,axis=1,inplace=True)
    #len(features_all_missing)
    df_missing=show_missing(df_data)
    
    def find_cols_with_missing_thres(df_missing,df_data,thres):
        df_missing=df_missing[df_missing['missing_count']>= thres*len(df_data)]
        cols=df_missing['col_name'].tolist()
        return cols
    cols_to_drop=find_cols_with_missing_thres(df_missing,df_data,missing_thres)
    
    df_data.drop(cols_to_drop,axis=1,inplace=True)
    
    features=list(df_data)
    
    df_catOrNum = df_catOrNum[df_catOrNum.Feature.isin(features)]
    
    features_categorical=df_catOrNum[df_catOrNum['Feature_type']=='Categorical']['Feature'].tolist()
    features_numerical=df_catOrNum[df_catOrNum['Feature_type']=='Numerical']['Feature'].tolist()
    
    X=df_data.copy()

    try:
        for f in features_categorical:
            X.loc[X[f].isnull(),f] = 'ND'

    except:
        pass




    df_missing=show_missing(X)

    try:
        for f in features_numerical:
            X.loc[X[f].isin(["C","M","H","T","__","","P","E","F","G","K","I"]),f] = np.nan
            X[f] = X[f].astype(float)

    except:
        pass
    
    
    features_all_missing1 = X.loc[:,X.notnull().sum()==0].columns
    
    X.drop(features_all_missing1,axis=1,inplace=True)

    features_numerical=list(set(features_numerical)-set(features_all_missing1))
    a = X[features_numerical].isnull().sum()
    
    dict_numerical_missing = {i:j for i,j in a.iteritems() if j>0}
    features_numerical_missing = list(dict_numerical_missing.keys())
    features_numerical_nonmissing = list(set(features_numerical)-set(features_numerical_missing))

    imputer = Imputer(strategy="median",axis=0)

    try:
        X_median = pd.DataFrame(imputer.fit_transform(X[features_numerical_missing])
                         ,columns=[i+'_Median' for i in features_numerical_missing]
                         ,index=X.index)

        X = X.merge(X_median,left_index=True,right_index=True)

    except:
        pass

    for f in features_numerical_missing:
        X[f+'_Zero'] = X[f].fillna(0)
        
        
    y = df_data[target]
    del X[target]
    dict_auc_bin = {i:0 for i in features_numerical_missing}
    dict_auc_median = {i:0 for i in features_numerical_missing}
    dict_auc_zero = {i:0 for i in features_numerical_missing}
    dict_auc_countND = {i:0 for i in features_numerical_missing}
    dict_auc_countBins = {i:0 for i in features_numerical_missing}
    for count,i in enumerate(features_numerical_missing):
        #print(str(count)+' out of '+str(len(features_numerical_missing)))
        #X_temp_bin = pd.get_dummies(X[i+'_Bin'])
        X_temp_median = X[i+'_Median']
        X_temp_zero = X[i].fillna(0)
        y_temp = y

        #### median ####
        clf = linear_model.LogisticRegression()
        clf.fit(X_temp_median.values.reshape(-1, 1),y_temp)
        y_score = list(map(lambda x: x[1],clf.predict_proba(X_temp_median.values.reshape(-1, 1))))
        dict_auc_median[i] = roc_auc_score(y_temp,y_score)
        #### zero ####
        clf = linear_model.LogisticRegression()
        clf.fit(X_temp_zero.values.reshape(-1, 1),y_temp)
        y_score = list(map(lambda x: x[1],clf.predict_proba(X_temp_zero.values.reshape(-1, 1))))
        dict_auc_zero[i] = roc_auc_score(y_temp,y_score)
    df_auc = pd.DataFrame({'AUC_bin':dict_auc_bin,'AUC_median':dict_auc_median,
                       'AUC_zero':dict_auc_zero,'NoMissing':dict_auc_countND,'NoBins':dict_auc_countBins}) 
    
    df_auc['AUC_max'] = ['Median' if j>z else 'Zero' for j,z in zip(df_auc['AUC_median'],df_auc['AUC_zero'])]


    try:

        median_list = list(df_auc.loc[df_auc.AUC_max=='Median',:].index)

    except:
        pass


    try:

        zero_list = list(df_auc.loc[df_auc.AUC_max=='Zero',:].index)

    except:
        pass
    #bin_list = list(df_auc.loc[df_auc.AUC_max=='Bin',:].index)
    features_numerical_missing_filled = [i+'_Median' for i in median_list] + [i+'_Zero' for i in zero_list]
    
    median_series = X[median_list].median()
    zeros = pd.Series({i:0 for i in zero_list})
    
    features=features_categorical + features_numerical_nonmissing + features_numerical_missing_filled
    #features = features =list(X)

    #try:

        #X.loc[:,'ResidentInUKMonths_Bin5'] = pd.qcut(X.loc[:,'ResidentInUKMonths'],q=5,duplicates='drop').astype("object")
        #del X['ResidentInUKMonths']

        #features.remove('ResidentInUKMonths')

    #except:
        #pass

    X_dummy = pd.get_dummies(X[features])
    features_dummy = X_dummy.columns

    features_dummy = X_dummy.columns
    
    
    def removePerfectCollinearity(X,features):
        rho = X[features].corr()
        rho = rho.where(np.triu(np.ones(rho.shape)).astype(np.bool))
        rho1 = rho.unstack()
        rho2 = rho1.reset_index().rename(columns={0:'Corr'})
        rho2 = rho2[rho2.level_0!=rho2.level_1].sort_values(by=['Corr','level_0','level_1'],ascending=False)
        features_duplicate = rho2[rho2.Corr==1].level_1.drop_duplicates()
        features_new = list(set(features) - set(features_duplicate))
        return(features_new)

    features_dummy = removePerfectCollinearity(X_dummy,features_dummy)

    X_dummy = X_dummy[features_dummy]

    #X_dummy.filter(regex="^ResidentInUKMonths").columns
    
    
    

    '''clf = RandomForestClassifier(n_estimators=500,min_samples_leaf=20,random_state=123)
    clf.fit(X_dummy,y)
    df_feat_importance = pd.DataFrame(sorted(zip(clf.feature_importances_,X_dummy.columns),reverse=True))
    df_feat_importance.columns = ['Importance','Feature']
    df_feat_importance.sort_values(by='Importance',ascending=False,inplace=True)

    features_filtered = list(df_feat_importance.loc[df_feat_importance.Importance>0,:].Feature.values)
    '''

    X_final = X_dummy#[features_filtered]

   

    features_numerical = X_final.dtypes[X_final.dtypes.isin([np.dtype('int64'),np.dtype('float64')])].index

    a = X_final[features_numerical].apply(lambda x: pd.Series({'k':kurtosis(x),'nonNegative':all(x>=0)})).T
    features_to_log = a.loc[(a.k>0)&(a.nonNegative),:].index



    for f in features_to_log:
        X_final.loc[:,f+'_log'] = np.log10(X_final.loc[:,f] + 1)
    X_final.drop(features_to_log,axis=1,inplace=True)

    a = X_final.nunique()
    features_binary = a[a==2].index
    features_nonbinary = list(set(X_final.columns)-set(features_binary))

    standard_df = pd.concat([X_final[features_nonbinary].mean(),X_final[features_nonbinary].std()],axis=1,keys=['Mean','Std'])

    if standardisation_needed ==1 :

        scaler = StandardScaler()
        X_final1 = pd.DataFrame(scaler.fit_transform(X_final[features_nonbinary]),columns=X_final[features_nonbinary].columns,index=X_final[features_nonbinary].index)
        X_final2 = X_final[features_binary]
        X_final = X_final1.merge(X_final2,right_index=True,left_index=True)

    #X_final.to_csv('X_features_digital_EQ3_1.csv')

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H_%M_%S')
    st=str(st).replace(" ", "_")

    X_new=X_final.copy()
    
    df_median=pd.DataFrame({'orig_feature':median_series.index, 'mis_val_imp':median_series.values})
    df_median['missImpute_Feature']=df_median['orig_feature'].astype(str)+'_Median'


    df_zero=pd.DataFrame({'orig_feature':zeros.index, 'mis_val_imp':zeros.values})
    df_zero['missImpute_Feature']=df_zero['orig_feature'].astype(str)+'_Zero'


    if save_data_scoring==1:
        pd.Series(features_to_log).to_csv('featuresToLog'+'_'+st+'_'+str(model_name)+'.csv')
        df_median.append(df_zero).to_csv('fill_missing_feats_median_zero_series'+'_'+st+'_'+model_name+'.csv')
        median_series.to_csv('fill_missing_feats_median'+'_'+st+'_'+model_name+'.csv')
        zeros.to_csv('fill_missing_feats_zeros_series2_v2'+'_'+st+'_'+model_name+'.csv')
        X_new.to_csv('CleanedData'+'_'+st+'_'+model_name+'.csv')
    if save_data_scoring==1 and standardisation_needed ==1:

        standard_df.to_csv('standardising'+'_'+st+'_'+model_name+'.csv')
        
        
    return X_new


