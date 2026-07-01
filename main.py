import numpy as np
import pandas as pd 
import joblib
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,mean_squared_error,r2_score

MODEL_FILE ='model.pkl'
PIPELINE_FILE ='pipeline.pkl'

def build_pipeline(num_att,cat_att):

    num_pipeline = Pipeline([
        ('impute',SimpleImputer(strategy='median')),
        ('SS',StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('OHE',OneHotEncoder(handle_unknown='ignore'))
    ])
    full_pipeline = ColumnTransformer([
        ('num',num_pipeline,num_att),
        ('cat',cat_pipeline,cat_att)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv('housing.csv')
    df['income_cat'] = pd.cut(df['median_income'],bins=[0,2,3,4,5,np.inf],labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(df,df['income_cat']):
        housing_train = df.loc[train_index].drop('income_cat',axis=1)
        housing_test = df.loc[test_index].drop('income_cat',axis=1)

#Features and labels
    housing = housing_train.drop('median_house_value',axis=1).copy()
    housing_labels = housing_train['median_house_value']

    num_att = housing.drop('ocean_proximity',axis=1).columns.tolist()
    cat_att =['ocean_proximity']

    pipeline = build_pipeline(num_att,cat_att)
    X = pipeline.fit_transform(housing)

    model = RandomForestRegressor(random_state=42)
    model.fit(X,housing_labels)

    Y_train= housing_test.drop('median_house_value',axis=1)
    Y_test = housing_test['median_house_value'].copy()

    Y = pipeline.transform(Y_train)
    Y_predictions = model.predict(Y)
    rmse = root_mean_squared_error(Y_test,Y_predictions)
    mse = mean_squared_error(Y_test,Y_predictions)
    r2 = r2_score(Y_test,Y_predictions)
    print('RMSE of model is:',round(rmse,2))
    print('MSE of model is:',round(mse,2))
    print('R2_SCORE of model is:',round(r2,4))

    #Save model and pipeline
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print('Model trained and saved')

else:
    model = joblib.load('model.pkl')
    pipeline = joblib.load('pipeline.pkl')
    input_data = pd.read_csv('input_csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions
    input_data.to_csv('output.csv',index=False)
    print('Inference completed.Results saved to output.csv')

