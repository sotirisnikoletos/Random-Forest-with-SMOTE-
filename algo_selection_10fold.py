import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class ml_models:
    """
    After obtaining the drug-target-disease by combining TTD and Knowldge Graph, we could adopt any machine learning method to train the model.
    Here, we tried logistic regression, decision tree, adaboost and xgboost. The logistic regression model is used as one of the baseline method.
    """
    def __init__(self, train_data_dir,output_dir):
        self.train_data_dir=train_data_dir
        self.output_dir = output_dir

    def cross_validation_for_model_selection(self):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ##
        # Set the under sampling  methods
        undersample = RandomUnderSampler(sampling_strategy=1, random_state=5)

        # Set the hyper parameters for logistic regression with different hyperparameters
        models_and_parameters = {
            "DecisionTree": {
                "model": DecisionTreeClassifier(),
                "tuned_parameters": [{'criterion': ['gini', 'entropy']}]
            },

            "AdaBoost": {
                "model": AdaBoostClassifier(),
                "tuned_parameters": [
                    {
                        'n_estimators': [25, 50, 100, 200],
                        'algorithm': ['SAMME']
                    }]
            },

            "XGBoost":{
                "model":XGBClassifier(missing=-1,use_label_encoder=False),
                "tuned_parameters": [
                    {
                    'n_estimators': [100, 200],
                    'max_depth':[3, 4, 5, 6, 7, 8],
                    'learning_rate':[ 0.01, 0.05, 0.1, 0.2, 0.3],
                    'subsample':[0.75,1]
                    }]
            }
            #"LogisticRegression": {
            #    "model": LogisticRegression(max_iter=200),
            #    "tuned_parameters": [
            #        {
            #            'penalty': ['elasticnet'],
            #            'solver': ['saga'],
            #            'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            #        }]

            #}

        }


        # Load data from data file
        
        data = pd.read_csv("shuffled_combined_file_no_0_2024.csv")

        # Randomly select some negative data as training data in order to balance the positive and negative data.
        # In general, undersampling and oversampling methods are only can be used after splitting training, validation and teting set
        # in oder to avoid overfitting and data leakage problem. Since here we use random undersampling method, we can directly use it here.
        feature_cols=["rel1_ADMINISTERED_TO", "rel1_AFFECTS", "rel1_ASSOCIATED_WITH", "rel1_AUGMENTS", "rel1_CAUSES", "rel1_COEXISTS_WITH", "rel1_compared_with", "rel1_COMPLICATES", "rel1_CONVERTS_TO", "rel1_DIAGNOSES", "rel1_different_from", "rel1_different_than", "rel1_DISRUPTS", "rel1_higher_than", "rel1_INHIBITS", "rel1_INTERACTS_WITH", "rel1_IS_A", "rel1_ISA", "rel1_LOCATION_OF", "rel1_lower_than", "rel1_MANIFESTATION_OF", "rel1_METHOD_OF", "rel1_OCCURS_IN", "rel1_PART_OF", "rel1_PRECEDES", "rel1_PREDISPOSES", "rel1_PREVENTS", "rel1_PROCESS_OF", "rel1_PRODUCES", "rel1_same_as", "rel1_STIMULATES", "rel1_TREATS", "rel1_USES", "rel1_MENTIONED_IN", "rel1_HAS_MESH", "rel2_ADMINISTERED_TO", "rel2_AFFECTS", "rel2_ASSOCIATED_WITH", "rel2_AUGMENTS", "rel2_CAUSES", "rel2_COEXISTS_WITH", "rel2_compared_with", "rel2_COMPLICATES", "rel2_CONVERTS_TO", "rel2_DIAGNOSES", "rel2_different_from", "rel2_different_than", "rel2_DISRUPTS", "rel2_higher_than", "rel2_INHIBITS", "rel2_INTERACTS_WITH", "rel2_IS_A", "rel2_ISA", "rel2_LOCATION_OF", "rel2_lower_than", "rel2_MANIFESTATION_OF", "rel2_METHOD_OF", "rel2_OCCURS_IN", "rel2_PART_OF", "rel2_PRECEDES", "rel2_PREDISPOSES", "rel2_PREVENTS", "rel2_PROCESS_OF", "rel2_PRODUCES", "rel2_same_as", "rel2_STIMULATES", "rel2_TREATS", "rel2_USES", "rel2_MENTIONED_IN", "rel2_HAS_MESH", "rel3_ADMINISTERED_TO", "rel3_AFFECTS", "rel3_ASSOCIATED_WITH", "rel3_AUGMENTS", "rel3_CAUSES", "rel3_COEXISTS_WITH", "rel3_compared_with", "rel3_COMPLICATES", "rel3_CONVERTS_TO", "rel3_DIAGNOSES", "rel3_different_from", "rel3_different_than", "rel3_DISRUPTS", "rel3_higher_than", "rel3_INHIBITS", "rel3_INTERACTS_WITH", "rel3_IS_A", "rel3_ISA", "rel3_LOCATION_OF", "rel3_lower_than", "rel3_MANIFESTATION_OF", "rel3_METHOD_OF", "rel3_OCCURS_IN", "rel3_PART_OF", "rel3_PRECEDES", "rel3_PREDISPOSES", "rel3_PREVENTS", "rel3_PROCESS_OF", "rel3_PRODUCES", "rel3_same_as", "rel3_STIMULATES", "rel3_TREATS", "rel3_USES", "rel3_MENTIONED_IN", "rel3_HAS_MESH"]
        X=data[feature_cols]
        y=data["INTERACTS"]
        X, y = undersample.fit_resample(X, y)
        X_all = np.asarray(X, dtype=np.float32)
        y_all = np.asarray(y)

        # seed could be any random integer
        seed=6

        # Split the data (X_all and y_all here) into training and validation set. 70% data are used as training data and 30% data are validation set.
        validation_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify=y_all , test_size=validation_size)

        # Select the best hyper-meters from different over sampling and machine learning methods by grid search
        for ele in models_and_parameters:
            # Select the best hyper-parameters by 10 cross validation
            clf = GridSearchCV(models_and_parameters[ele]["model"], models_and_parameters[ele]["tuned_parameters"],
                               scoring="roc_auc", cv=10)

            # Train the model with the best hyper parameters. Using validation set to early stop the training epochs.
            model = clf.fit(X_train, y_train).best_estimator_

            early_stopping_rounds = 20
            best_auc = 0
            iter_train = 0
            while iter_train < early_stopping_rounds:
                model, auc_score = self.fit_and_score(model, X_train, y_train, X_test, y_test)
                if best_auc < auc_score:
                    best_auc = auc_score
                    iter_train = 0
                else:
                    iter_train += 1

            ##------------ statistic start----------------
            # test the model
            predict_label = model.predict(X_test)
            predict_prob = model.predict_proba(X_test)[:, 1]
            FP, FN, TP, TN = 0

            for i in range(len(predict_label)):
                if predict_label[i] == 1 and y_test[i] == 1:
                    TP += 1
                if predict_label[i] == 1 and y_test[i] == 0:
                    FP += 1
                if predict_label[i] == 0 and y_test[i] == 1:
                    FN += 1
                if predict_label[i] == 0 and y_test[i] == 0:
                    TN += 1


            if (TP + FP) == 0:
                PPV = 0
            else:
                PPV = TP / (TP + FP)


            if (TP + FP + FN + TN) == 0:
                ACC = 0
            else:
                ACC = (TP + TN) / (TP + FP + FN + TN)


            if (TN + FP) == 0:
                Specificity = 0
            else:
                Specificity = TN / (TN + FP)

            fpr, tpr, _ = roc_curve(y_test, predict_prob)
            AUC = auc(fpr, tpr)
            ACC = (TP + TN) / (TP + TN + FP + FN)
            Sensitivity = (TP) / (TP + FN)
            Specificity = (TN) / (TN + FP)
            PPV = TP / (TP + FP)
            # NPV
            if (TN + FN) == 0:
                NPV = 0
            else:
                NPV = TN / (TN + FN)
            Fscore = (2 * PPV * Sensitivity) / (PPV + Sensitivity)

            print("========================================RESULTS========================================")
            print("%+10s\t%+10s\t%+10s\t%+10s\t%+10s\t%+10s\t%+10s" % (
            "AUC", "ACC", "PPV", "Sensitivity", "F-score", "Specificity", "NPV"))
            print("%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f" % (
            AUC, ACC, PPV, Sensitivity, Fscore, Specificity, NPV))
            print("========================================================================================")

        pickle.dump(model, open(output_dir + "smartHEALTH_model", "wb+"))

    def fit_and_score(self, estimator, X_train, y_train, X_test, y_test):
        '''Fit the estimator on the train set and score it on test set'''
        estimator.fit(X_train, y_train)
        y_score = estimator.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_score)

        return estimator, auc_score


if __name__ == "__main__":
    train_data_dir="/home/snikoletos/"
    output_dir="/home/snikoletos/"

    s=ml_models(train_data_dir, output_dir)
    s.cross_validation_for_model_selection()
