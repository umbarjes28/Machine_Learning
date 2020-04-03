# Implementing Multiclassifier decision tree for Question 2

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz



import pandas as pd

# loading the iris dataset
#iris = datasets.load_iris()
balance_data = pd.read_csv("iris.data.csv", sep=',', header=None)
X = balance_data.values[:, 0:3]
Y = balance_data.values[:, 4]
# X -> features, y -> label
#X = iris.data
#y = iris.target

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)

#calculating precision and recall
true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos
false_neg = np.sum(cm, axis=1) - true_pos
true_neg = np.sum(cm) - (false_pos + false_neg + true_pos)

precision = np.sum(true_pos / (true_pos + false_pos))
recall = np.sum(true_pos / (true_pos + false_neg))

sensitivity = true_pos / (true_pos + false_pos)
specificity = true_neg / (false_pos + true_neg)

#calculating accuracy
accuracy= accuracy_score(y_test, dtree_predictions)

#classification report
cr = classification_report(y_test, dtree_predictions)

#Roc curve
plt.plot(y_test, dtree_predictions, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

#Visualize tree
dot_data = StringIO()
export_graphviz(dtree_model, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())

print(dtree_predictions)
print('---------------- confusion matrix ------------')
print(cm)
print('---------------- Accuracy ------------')
print(accuracy)
print('---------------- classification report ------------')
print(cr)
print('---------------- precision ------------')
print(precision)
print('---------------- recall ------------')
print(recall)
print('---------------- sensitivity ------------')
print(sensitivity)
print('---------------- specificity ------------')
print(specificity)
