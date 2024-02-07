import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from collections import Counter



#FOR PRINT AND VISUALS


model = list()
resample = list()
precision = list()
recall = list()
F1score = list()
AUCROC = list()
def test_eval(clf_model, X_test, y_test, algo=None, sampling=None):
    # Test set prediction
    y_prob=clf_model.predict_proba(X_test)
    y_pred=clf_model.predict(X_test)

    print('Confusion Matrix')
    print('='*60)
    print(confusion_matrix(y_test,y_pred),"\n")
    print('Classification Report')
    print('='*60)
    print(classification_report(y_test,y_pred),"\n")
    print('AUC-ROC')
    print('='*60)
    print(roc_auc_score(y_test, y_prob[:,1]))
          
    model.append(algo)
    precision.append(precision_score(y_test,y_pred))
    recall.append(recall_score(y_test,y_pred))
    F1score.append(f1_score(y_test,y_pred))
    AUCROC.append(roc_auc_score(y_test, y_prob[:,1]))
    resample.append(sampling)







data = pd.read_csv("shuffled_combined_file_no_0_2024.csv")

feature_cols=["rel1_ADMINISTERED_TO", "rel1_AFFECTS", "rel1_ASSOCIATED_WITH", "rel1_AUGMENTS", "rel1_CAUSES", "rel1_COEXISTS_WITH", "rel1_compared_with", "rel1_COMPLICATES", "rel1_CONVERTS_TO", "rel1_DIAGNOSES", "rel1_different_from", "rel1_different_than", "rel1_DISRUPTS", "rel1_higher_than", "rel1_INHIBITS", "rel1_INTERACTS_WITH", "rel1_IS_A", "rel1_ISA", "rel1_LOCATION_OF", "rel1_lower_than", "rel1_MANIFESTATION_OF", "rel1_METHOD_OF", "rel1_OCCURS_IN", "rel1_PART_OF", "rel1_PRECEDES", "rel1_PREDISPOSES", "rel1_PREVENTS", "rel1_PROCESS_OF", "rel1_PRODUCES", "rel1_same_as", "rel1_STIMULATES", "rel1_TREATS", "rel1_USES", "rel1_MENTIONED_IN", "rel1_HAS_MESH", "rel2_ADMINISTERED_TO", "rel2_AFFECTS", "rel2_ASSOCIATED_WITH", "rel2_AUGMENTS", "rel2_CAUSES", "rel2_COEXISTS_WITH", "rel2_compared_with", "rel2_COMPLICATES", "rel2_CONVERTS_TO", "rel2_DIAGNOSES", "rel2_different_from", "rel2_different_than", "rel2_DISRUPTS", "rel2_higher_than", "rel2_INHIBITS", "rel2_INTERACTS_WITH", "rel2_IS_A", "rel2_ISA", "rel2_LOCATION_OF", "rel2_lower_than", "rel2_MANIFESTATION_OF", "rel2_METHOD_OF", "rel2_OCCURS_IN", "rel2_PART_OF", "rel2_PRECEDES", "rel2_PREDISPOSES", "rel2_PREVENTS", "rel2_PROCESS_OF", "rel2_PRODUCES", "rel2_same_as", "rel2_STIMULATES", "rel2_TREATS", "rel2_USES", "rel2_MENTIONED_IN", "rel2_HAS_MESH", "rel3_ADMINISTERED_TO", "rel3_AFFECTS", "rel3_ASSOCIATED_WITH", "rel3_AUGMENTS", "rel3_CAUSES", "rel3_COEXISTS_WITH", "rel3_compared_with", "rel3_COMPLICATES", "rel3_CONVERTS_TO", "rel3_DIAGNOSES", "rel3_different_from", "rel3_different_than", "rel3_DISRUPTS", "rel3_higher_than", "rel3_INHIBITS", "rel3_INTERACTS_WITH", "rel3_IS_A", "rel3_ISA", "rel3_LOCATION_OF", "rel3_lower_than", "rel3_MANIFESTATION_OF", "rel3_METHOD_OF", "rel3_OCCURS_IN", "rel3_PART_OF", "rel3_PRECEDES", "rel3_PREDISPOSES", "rel3_PREVENTS", "rel3_PROCESS_OF", "rel3_PRODUCES", "rel3_same_as", "rel3_STIMULATES", "rel3_TREATS", "rel3_USES", "rel3_MENTIONED_IN", "rel3_HAS_MESH"]
X=data[feature_cols]
y=data["INTERACTS"]
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



#SIMPLE RANDOM FOREST + GRID

estimators = [2,10,30,50,100]
max_depth = [i for i in range(5,16,2)]
min_samples_split = [2, 5, 10, 15, 20, 50, 100]
min_samples_leaf = [1, 2, 5]
rf_model = RandomForestClassifier()

rf_params={'n_estimators':estimators,
           'max_depth':max_depth,
           'min_samples_split':min_samples_split}


cv = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)

clf_RF = RandomizedSearchCV(rf_model, rf_params, cv=cv, scoring='roc_auc', n_jobs=-1, n_iter=20, verbose=2)
clf_RF.fit(trainX, trainy)
print(clf_RF.best_estimator_)
test_eval(clf_RF, testX, testy, 'Random Forest', 'actual')




#SMOTE
from imblearn.over_sampling import SMOTE

smt = SMOTE()
#X_train, y_train = smt.fit_resample(X_train, y_train)
X_train_sm, y_train_sm = smt.fit_resample(trainX, trainy)
clf_RF.fit(X_train_sm, y_train_sm)
print(clf_RF.best_estimator_)
test_eval(clf_RF, testX, testy, 'Random Forest', 'smote')





#SMOTE ENN
from imblearn.combine import SMOTEENN

smenn = SMOTEENN()
X_train_smenn, y_train_smenn = smenn.fit_resample(trainX, trainy)
clf_RF.fit(X_train_smenn, y_train_smenn)
print(clf_RF.best_estimator_)
test_eval(clf_RF, testX, testy, 'Random Forest', 'smote+enn')

