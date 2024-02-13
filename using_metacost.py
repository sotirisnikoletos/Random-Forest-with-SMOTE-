from sklearn.metrics import precision_score, recall_score, PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from scipy.spatial import distance_matrix
from collections import Counter
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, fbeta_score
import seaborn as sns
import numpy as np
import pandas as pd

data = pd.read_csv("shuffled_combined_file_no_0_2024.csv")

feature_cols=["rel1_ADMINISTERED_TO", "rel1_AFFECTS", "rel1_ASSOCIATED_WITH", "rel1_AUGMENTS", "rel1_CAUSES", "rel1_COEXISTS_WITH", "rel1_compared_with", "rel1_COMPLICATES", "rel1_CONVERTS_TO", "rel1_DIAGNOSES", "rel1_different_from", "rel1_different_than", "rel1_DISRUPTS", "rel1_higher_than", "rel1_INHIBITS", "rel1_INTERACTS_WITH", "rel1_IS_A", "rel1_ISA", "rel1_LOCATION_OF", "rel1_lower_than", "rel1_MANIFESTATION_OF", "rel1_METHOD_OF", "rel1_OCCURS_IN", "rel1_PART_OF", "rel1_PRECEDES", "rel1_PREDISPOSES", "rel1_PREVENTS", "rel1_PROCESS_OF", "rel1_PRODUCES", "rel1_same_as", "rel1_STIMULATES", "rel1_TREATS", "rel1_USES", "rel1_MENTIONED_IN", "rel1_HAS_MESH", "rel2_ADMINISTERED_TO", "rel2_AFFECTS", "rel2_ASSOCIATED_WITH", "rel2_AUGMENTS", "rel2_CAUSES", "rel2_COEXISTS_WITH", "rel2_compared_with", "rel2_COMPLICATES", "rel2_CONVERTS_TO", "rel2_DIAGNOSES", "rel2_different_from", "rel2_different_than", "rel2_DISRUPTS", "rel2_higher_than", "rel2_INHIBITS", "rel2_INTERACTS_WITH", "rel2_IS_A", "rel2_ISA", "rel2_LOCATION_OF", "rel2_lower_than", "rel2_MANIFESTATION_OF", "rel2_METHOD_OF", "rel2_OCCURS_IN", "rel2_PART_OF", "rel2_PRECEDES", "rel2_PREDISPOSES", "rel2_PREVENTS", "rel2_PROCESS_OF", "rel2_PRODUCES", "rel2_same_as", "rel2_STIMULATES", "rel2_TREATS", "rel2_USES", "rel2_MENTIONED_IN", "rel2_HAS_MESH", "rel3_ADMINISTERED_TO", "rel3_AFFECTS", "rel3_ASSOCIATED_WITH", "rel3_AUGMENTS", "rel3_CAUSES", "rel3_COEXISTS_WITH", "rel3_compared_with", "rel3_COMPLICATES", "rel3_CONVERTS_TO", "rel3_DIAGNOSES", "rel3_different_from", "rel3_different_than", "rel3_DISRUPTS", "rel3_higher_than", "rel3_INHIBITS", "rel3_INTERACTS_WITH", "rel3_IS_A", "rel3_ISA", "rel3_LOCATION_OF", "rel3_lower_than", "rel3_MANIFESTATION_OF", "rel3_METHOD_OF", "rel3_OCCURS_IN", "rel3_PART_OF", "rel3_PRECEDES", "rel3_PREDISPOSES", "rel3_PREVENTS", "rel3_PROCESS_OF", "rel3_PRODUCES", "rel3_same_as", "rel3_STIMULATES", "rel3_TREATS", "rel3_USES", "rel3_MENTIONED_IN", "rel3_HAS_MESH"]
X=data[feature_cols]
y=data["INTERACTS"]
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

class MetaCost:
   

    """
    +-----------------+--------------+--------------+
    |                 |         Actual class        |
    + Predicted Class +--------------+--------------+
    |                 | class0       | class1       |
    +-----------------+--------------+--------------+
    | h(x) = class0   |      0       |      a       |
    | h(x) = class1   |      c       |      0       |
    +-----------------+--------------+--------------+
    | C = np.array([[0, a],[c, 0]])                 |
    +-----------------------------------------------+

    """
    # def __init__(self, classifier, cost_matrix, resamples=10, fraction=1, num_class=2):
    def __init__(self, classifier, cost_matrix, resamples=50, fraction=1, num_class=2):
        self.classifier = classifier
        self.cost_matrix = cost_matrix
        self.resamples = resamples
        self.fraction = fraction
        self.num_classes = num_class



   

   

    def plot_dataset(X, y):
        cmap_bold = ["darkorange", "darkblue"]
        markers = {1: "X", 0: "v"}
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=y,
            palette=cmap_bold,
            alpha=1.0,
            edgecolor="black",
            style=y, markers=markers
        )
    
    def plot_decision_boundary(self,X, y, clf, clf_name):
        _, ax = plt.subplots()
        cmap_light = ListedColormap(["orange", "white"])
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            cmap=cmap_light,
            ax=ax,
            response_method="predict",
            alpha=0.5,
            xlabel='feature_1',
            ylabel='feature_2',
        )
        disp.plot(plot_method="contour", cmap="gist_gray", ax = ax, alpha = 1.)
        # Plot also the training points
        self.plot_dataset(X, y)
        plt.title("Classification using %s classifier" %clf_name)


    def fit(self, X, y):
        # Combine the features and the target into one DataFrame
        data = pd.DataFrame(data=X)
        data['target'] = y

        # Calculate the number of samples in each resample
        num_samples = len(data) * self.fraction

        # Resample the data, train a model on each resample, and store the models
        models = [self._train_on_resample(data.sample(n=int(num_samples), replace=True)) for _ in range(self.resamples)]

        # Relabel the instances based on the models' predictions and the cost matrix
        new_labels = self._relabel_instances(data, models)

        # Train a new model on the data with relabeled instances
        final_model = clone(self.classifier)
        final_model.fit(data.drop(columns=['target']).values, new_labels)

        self.model = final_model
        self.__class__ = self.model.__class__
        self.__dict__ = self.model.__dict__
        return final_model

    def _train_on_resample(self, resample):
        X_resampled = resample.drop(columns=['target']).values
        y_resampled = resample['target'].values
        model = clone(self.classifier)
        return model.fit(X_resampled, y_resampled)

    def _relabel_instances(self, data, models):
        X_array = data.drop(columns=['target']).values
        labels = []
        for i in range(len(data)):
            class_probs = [self._get_class_probs(model, X_array[[i]]) for model in models]
            average_probs = np.mean(class_probs, 0).T
            labels.append(self._get_new_label(average_probs))
        return labels

    def _get_class_probs(self, model, instance):
        return model.predict_proba(instance)

    def _get_new_label(self, average_probs):
        # Multiply the cost matrix with the average probabilities.
        # This gives us the expected costs for each class.
        costs = self.cost_matrix.dot(average_probs)

        # Find the class with the minimum expected cost.
        # This is the new label for the instance.
        return np.argmin(costs)
    
    def compute_scores(model, X_test, y_test):
            y_pred = model.predict(X_test)
            f2 = fbeta_score(y_test, y_pred, beta=2.)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            return f2, precision, recall
    
    RF_clf = RandomForestClassifier(random_state=0).fit(trainX, trainy)

    plot_decision_boundary(trainX, trainy, RF_clf, 'RandomForestClassifier')
    plt.show()

    print(classification_report_imbalanced(testy, RF_clf.predict(testX), target_names=['class 0', 'class 1']))

    f2, precision_val, recall_val =  compute_scores(RF_clf, testX, testy)

    print(f"f2-score: {f2:.3f}", f"precision: {precision_val:.3f}", f"recall: {recall_val:.3f}")

    display = PrecisionRecallDisplay.from_estimator(RF_clf, testX, testy, ax = plt.gca(),name = "", pos_label=1)
    display.ax_.set_title("Classification using RandomForestClassifier classifier")
    y_pred = RF_clf.predict(testX)
    precision, recall, _ = precision_recall_curve(testy, y_pred)
    f2_scores = 5 * recall * precision / (recall + 4*precision)
    plt.plot(recall[np.argmax(f2_scores)], precision[np.argmax(f2_scores)], marker='x', color= "red", label=' (f2-score = '+ str(round(f2,3))+')')
    plt.plot([], [], ' ', label=' (precision= ' + str(round(precision_val, 3)) + ')')
    plt.plot([], [], ' ', label=' (recall= ' + str(round(recall_val, 3)) + ')')
    plt.legend(loc= 'lower left', fontsize = 'xx-large')
    plt.ylabel("Precision", fontsize = 'xx-large')
    plt.xlabel("Recall", fontsize = 'xx-large')
