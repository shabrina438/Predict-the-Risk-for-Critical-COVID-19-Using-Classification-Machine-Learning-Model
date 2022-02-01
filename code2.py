%python
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import oml

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


%python
from sklearn import linear_model
import pandas as pd

Covid_data = oml.sync(query='SELECT * FROM COVID_DATA', schema = "CD")
Covid_data.head()

%python

Covid_data.shape

%python
DEMO_DF = Covid_data[["DAY", "LAB_TEST", "CONFIRMED", "OUTPUT"]]

%python
z.show(DEMO_DF.head())

%python
DEMO_DF = DEMO_DF.replace(old = ['LOW RISK','MEDIUM RISK','HIGH RISK'], new = [1.0,2.0,3.0], default = 0.0, columns = ['OUTPUT'])

%python
z.show(DEMO_DF.head())

%python

z.show(DEMO_DF.crosstab('OUTPUT'))

%python
TRAIN, TEST = DEMO_DF.split(ratio = (0.7,0.3))
TRAIN_X = TRAIN.drop('OUTPUT')
TRAIN_Y = TRAIN['OUTPUT']
TEST_X = TEST
TEST_Y = TEST['OUTPUT']

%python
try:
    oml.drop(model = 'DT_CLAS_MODEL') 
except:
    print("No such model")

settings = {'TREE_IMPURITY_METRIC': 'TREE_IMPURITY_GINI', 
            'TREE_TERM_MAX_DEPTH': '10',
            'TREE_TERM_MINPCT_NODE': '0.025', 
            'TREE_TERM_MINPCT_SPLIT': '0.05',
            'TREE_TERM_MINREC_NODE': '05',
            'TREE_TERM_MINREC_SPLIT': '10',
            'CLAS_MAX_SUP_BINS': '20'} 


dt_mod = oml.dt(**settings)
dt_mod.fit(TRAIN_X, TRAIN_Y, case_id = 'DAY', model_name = 'DT_CLAS_MODEL')

%python

dt_mod.score(TEST_X, TEST_Y)

%python

# Set the case ID attribute
case_id = 'DAY'
# Gather the Predictions
RES_DF = dt_mod.predict(TEST_X, supplemental_cols = TEST_X)
# Additionally collect the PROBABILITY
RES_PROB = dt_mod.predict_proba(TEST_X, supplemental_cols = TEST_X[case_id])
# Join the entire result into RES_DF
RES_DF = RES_DF.merge(RES_PROB, how = "inner", on = case_id, suffixes = ["", ""])


%python

z.show(RES_DF[ RES_DF.columns])

%python

z.show(RES_DF.crosstab(['OUTPUT','DAY','PREDICTION']))

%python
z.show(RES_DF.crosstab('DAY','PREDICTION'))

%python

RES_DF = dt_mod.predict(TEST_X, supplemental_cols = TEST_X[['DAY','OUTPUT']], topN_attrs = True)
z.show(RES_DF)

%python

try:
    oml.drop(model = 'RF_CLAS_MODEL') 
except:
    pass

settings = {'RFOR_MTRY': '3', 
            'RFOR_NUM_TREES': '70', 
            'RFOR_SAMPLING_RATIO': '0.25'}

rf_mod = oml.rf(**settings)
rf_mod.fit(TRAIN_X, TRAIN_Y, case_id = 'DAY', model_name = 'RF_CLAS_MODEL')

%python
rf_mod.score(TEST_X, TEST_Y)

%python
# Set the case ID attribute
case_id = 'DAY'
# Gather the Predictions
RES_DF = rf_mod.predict(TEST_X, supplemental_cols = TEST_X)
# Additionally collect the PROBABILITY_OF_0 and PROBABILITY_OF_1
RES_PROB = rf_mod.predict_proba(TEST_X, supplemental_cols = TEST_X[case_id])
# Join the entire result into RES_DF
RES_DF = RES_DF.merge(RES_PROB, how = "inner", on = case_id, suffixes = ["", ""])

%python

z.show(RES_DF[RES_DF.columns])

%python

z.show(RES_DF.crosstab(['OUTPUT','DAY','PREDICTION']))


%python

z.show(RES_DF.crosstab('DAY','PREDICTION'))

%python
RES_DF = dt_mod.predict(TEST_X, supplemental_cols = TEST_X[['DAY','OUTPUT']], topN_attrs = True)
z.show(RES_DF)