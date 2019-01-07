
'''
Important notes

1) This algorithm should be run after feature selection process
2) feature descriptions file should be in the working directory. Feature description should have the original feature name and the description of the feature.
Precise Column names should be [Feature,Description]


'''


'''
Input parameters

data : full dataset before splitting train and test sets

features_numerical (a list): numerical features should be in a list. Feature names should not be original feature names. They should be transformed names

target : speicify the name of the target variable

description_needed : set this 1 if you need the decriptions of the features

no_iterations : number of time you need to run this algorithm to identify features which has better predictive power with interacting with another feature

remove_cols : this should be a list. Put all the variables you need to remove before running the algo

train_testsplit_date : give the date where you need to split training and test. Performace after adding interations is measured on the test set

selected_features : selected features up to this stage


feature_descrption_filename : give the name of the description file as a string
'''





import datetime
import time
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
from xgboost import XGBClassifier


from sklearn import preprocessing
import random


def create_2_feature_interactions(data,features_numerical,target,decription_needed,no_iterations
                                  ,remove_cols,train_testsplit_date,selected_features,feature_descrption_filename) :
    



    def get_oroginal_feature(x):
    
    
        if 'log' in x:
            
            x=x[:-4]
            
            
            
            
            if 'Zero' in x:

                return x[:-5]

            elif 'Median' in x:
                return x[:-7]

            else:
                return x
            
            
        else:
            
            if 'Zero' in x:

                return x[:-5]

            elif 'Median' in x:
                return x[:-7]

            else:
                return x


    def get_description(x):
        df_description2=df_description[df_description['Feature']==x].reset_index(drop=True)

        return df_description2.iloc[0][1]



    
    #df_coef.loc[:,'Feature_Original'] = [i.split('_')[0] for i in df_coef.Feature]
    selected_feats=selected_features
  
        

    X_final_all2=data
    
    
    
    
    AUC_withOld=[]
    AUC_onlyNew=[]
    FEATS=[]
    feat1=[]
    feat2=[]
 


    for i in range(0,no_iterations):
        random_two=random.sample(set(features_numerical), 2)

        X_final_all2[random_two[0]+"$"+random_two[1]]=X_final_all2[random_two[0]] * X_final_all2[random_two[1]]

        selected_feats2=selected_feats+[(random_two[0]+"$"+random_two[1])]
        
        train=X_final_all2[X_final_all2['LoanApplicationDateKey']<train_testsplit_date ] #20180610 20180528
        test=X_final_all2[X_final_all2['LoanApplicationDateKey']>=train_testsplit_date ]
       
        for i in remove_cols:
            try:
                del train[i]
            except:
                continue
                
        for i in remove_cols:
                try:
                    del test[i]
                except:
                    continue
        train1=train.copy()
        test1=test.copy()
        
       
        y = train1[target]
        y_test = test1[target]

       


        del train1[target]
        del test1[target]

        

        lr_model =linear_model.LogisticRegression(class_weight={0: 1,1:1})##,
    

        lr_model.fit(train1[selected_feats2], y)


        import math 
        probclf=[]
        
        for a, b in lr_model.predict_proba(test1[selected_feats2]):
            probclf.append(b)

        val_pred_xgb_probxgb= probclf
        auc1 = roc_auc_score(y_test, val_pred_xgb_probxgb)

        FEATS.append(random_two[0]+"$"+random_two[1])
        AUC_withOld.append(auc1)



        train2=train.copy()
        test2=test.copy()


        y = train2[target]
        y_test = test2[target]

       


        del train2[target]
        del test2[target]
        
        try:

            selected_feats2.remove(random_two[0])

        except:
            pass

        try:
            selected_feats2.remove(random_two[1])

        except:
            pass

        lr_model =linear_model.LogisticRegression(class_weight={0: 1,1:1})##,
    # Extract the two most important features

        lr_model.fit(train2[selected_feats2], y)


        import math 
        probclf=[]
        for a, b in lr_model.predict_proba(test2[selected_feats2]):
            probclf.append(b)

        val_pred_xgb_probxgb= probclf
        auc2 = roc_auc_score(y_test, val_pred_xgb_probxgb)
        AUC_onlyNew.append(auc2)

        feat1.append(random_two[0])
        feat2.append(random_two[1])
        
        
    df_interactions=pd.DataFrame()
    df_interactions['new_feature']=FEATS
    df_interactions['AUC_withOld']=AUC_withOld
    df_interactions['AUC_onlyNew']=AUC_onlyNew
    df_interactions['feat1']=feat1
    df_interactions['feat2']=feat2
    
    df_interactions['Feature1_Original'] = df_interactions['feat1'].map(get_oroginal_feature)    
    df_interactions['Feature2_Original'] = df_interactions['feat2'].map(get_oroginal_feature)  
    
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H_%M_%S')
    st=str(st).replace(" ", "_")
    
    if decription_needed==1:
        
        df_description = pd.read_csv(feature_descrption_filename,encoding='ISO-8859-1')

        df_interactions['feat1_desc']=df_interactions['Feature1_Original'].map(get_description)  
        df_interactions['feat2_desc']=df_interactions['Feature2_Original'].map(get_description) 
    
    df_interactions.to_csv('interactions'+'_'+st+'_'+str(no_iterations)+'.csv')
    
    return df_interactions
    
