import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from datetime import date
import scipy.stats as stats
import math
from clean import clean_df
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import  roc_auc_score, roc_curve, auc,  accuracy_score,f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle




def optimize_model2_randomCV(model, grid_params, X_train, y_train, scoring):
    """[Takes a model in, grid parameters, X and y values returns  best model found using the input parameters]
    Args:
        model ([Classifcation model sklearn]): [Logistic Regression, Random Forrest, Gradient Boost, and others]
        grid_params ([dictionary]): [keys are strings of parater, values are list of values to try]
        X_train ([Pandas dataframe]): [training feature data]
        y_train ([numpy array]): [array of target values]
        scoring ([scoring type to measure]): [sklearn scoring options for given model]
    Returns:
        [type]: [description]
    """

    model_search = RandomizedSearchCV(model
                                        ,grid_params
                                        ,n_jobs=-1
                                        ,verbose=False
                                        ,scoring=scoring)
    model_search.fit(X_train, y_train)
    print(f"Best Parameters for {model}: {model_search.best_params_}")
    print(f"Best Model for {model}: {model_search.best_estimator_}")
    print(f"Best Score for {model}: {model_search.best_score_:.4f}")
    
    return model_search.best_estimator_

def best_model_predictor(model, X_test, y_test):
    """[returns Analysis of model on test set or valudation set]
    Args:
        model ([sklearn classifer model]): [Logistic regression, Random Forrest, Gradient Boosting, etc]
        X_test ([Pandas dataframe]): [Test feature data]
        y_test ([numpy array]): [target valudation data]
    """

    
    y_hats = model.predict(X_test)
    print(f"{model} ROC Score = {roc_auc_score(y_test, y_hats):.3f}")
    print(f"{model} F1 Score = {f1_score(y_test, y_hats):.3f}")
    print(f"{model} Accuracy Score = {accuracy_score(y_test, y_hats):.3f}")
    print(classification_report(y_test, y_hats))

def roc_curve_grapher(model, X_test ,y_test):
    """[Makes ROC curve graph given model and data]
    Args:
        model ([SKlearn classifer model]): [Logistic regression, Random Forrest, Gradient Boosting, etc]]
        X_test ([Pandas dataframe]): [Test feature data]
        y_test ([numpy array]): [target valudation data]
    """

    yhat = model.predict_proba(X_test)
    yhat = yhat[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, yhat)
    plt.plot([0,1], [0,1], linestyle='--', label='Random guess')
    plt.plot(fpr, tpr, marker='.', label=f'Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.suptitle('Model ROC curve', fontsize=20)
    plt.legend()
    
    plt.show()

def numerical_df_maker(df):
    """[Turns airBnB cleaned dataframe from clean_df into numerial for modeling]

    Args:
        df ([pandas dataframe]): [Pandas Dataframe cleaned from clean_df function]

    Returns:
        [dataframe]: [df numerical values for modeling]
    """

    # df5 = combined_df3.copy()
    
    col_list = ['id_guest_anon', 'id_host_anon', 'id_listing_anon','contact_channel_first','length_of_stay',
       'ts_interaction_first', 'ts_reply_at_first', 'ts_accepted_at_first','ds_checkin_first', 'ds_checkout_first','id_user_anon',
            'country','booked', 'date_interaction_first', 'response_time','listing_neighborhood']

    df.drop(col_list, axis = 1,inplace= True)
    d2 =  {
    'past_booker':1
    ,'new':0}
    df['guest_user_stage_first'].replace(d2, inplace= True)
    df = pd.get_dummies(df, prefix=['A'], columns=['room_type'])

    return df


if __name__ == "__main__":
    listings_df =pd.read_csv('~/Downloads/2018 DA Take Home Challenge/listings.csv')
    contacts_df =pd.read_csv('~/Downloads/2018 DA Take Home Challenge/contacts.csv')
    users_df = pd.read_csv('~/Downloads/2018 DA Take Home Challenge/users.csv')

    combined_df =contacts_df.merge(listings_df, left_on ='id_listing_anon', right_on='id_listing_anon')
    combined_df2 =combined_df.merge(users_df, left_on ='id_guest_anon', right_on='id_user_anon')
    combined_df3 = combined_df2.copy()

    contact_me_df3,instant_book_df3,book_it_df3 ,combined_df3 =  clean_df(combined_df3)

    # print(contact_me_df3)

    gradient_boosting_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
                         ,'max_depth': [2, 4, 8]
                         ,'subsample': [0.25, 0.5, 0.75, 1.0]
                         ,'min_samples_leaf': [1, 2, 4]
                         ,'max_features': ['sqrt', 'log2', None]
                         ,'n_estimators': [5,10,25,50,100,200]}

    random_forest_grid = {'max_depth': [2, 4, 8]
                        ,'max_features': ['sqrt', 'log2', None]
                        ,'min_samples_leaf': [1, 2, 4]
                        ,'min_samples_split': [2, 4]
                        ,'bootstrap': [True, False]
                        ,'class_weight': ['balanced']

                        ,'n_estimators': [5,10,25,50,100,200]}

    logistic_regression_grid = {'Cs':[2,5,10, 25, 100, 200]
                        ,'cv':[4]
                        ,'solver':['liblinear']#'lbfgs',
    #                        ,'max_iter' : [50]
                        ,'class_weight':['balanced']
                        ,'penalty':['l1'] #, 'l2', 'elasticnet'
                            
                         }
    logistic2_regression_grid = {'C':[0.05, 0.75, 0.1, 0.15, 0.3]
#                        ,'cv':[4]
                       ,'solver':['liblinear']#'lbfgs',
#                        ,'max_iter' : [50]
                       ,'class_weight':['balanced']
                       ,'penalty':['l1']} #, 'l2', 'elasticnet'


    
    # print(numerical_df_maker(contact_me_df3))

    df7 = numerical_df_maker(contact_me_df3)
    df7.dropna(inplace = True)
    print(df7.info())

    y = df7.pop('ts_booking_at')
    X = df7

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1, stratify = y)
    X_train2, X_test, y_train2, y_test = train_test_split(X_train.copy(), y_train.copy(), test_size=0.10, random_state=1)   

    print(X_test,y_test)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1, stratify = y)
    # X_train2, X_test, y_train2, y_test = train_test_split(X_train.copy(), y_train.copy(), test_size=0.10, random_state=1)

    results = optimize_model2_randomCV(LogisticRegression(), logistic2_regression_grid, X_train, y_train, scoring= 'roc_auc')
    print (results)
    

    best_model_predictor(results, X_test, y_test)
