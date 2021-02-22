from sklearn.model_selection import train_test_split, cross_val_score

from time import process_time
from typing import List

##Load a bunch of classification models for comparison
##Classification involves trade-offs
##Take into consideration time, accuracy, and explanability of model for benchmarking
from sklearn.linear_model import LogisticRegression #good base line against which to test more complex models.  
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier    #very powerful, but potentially time-intensive and impossible to explain details of the model
from sklearn.neighbors import KNeighborsClassifier  #intuitive visually
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class Benchmarker:

    #dictionary of models
    models = {  "Logistic Regression": LogisticRegression(),
                "Gaussian Naive Bayes": GaussianNB(),
                "Multinomial Naive Bayes": MultinomialNB(),
                "Nearest Neighbors": KNeighborsClassifier(3), 
                "Decision Tree": DecisionTreeClassifier(max_depth=10),
                "Random Forest": RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
                "Neural Net": MLPClassifier(alpha=1, max_iter=2000),
                "AdaBoost": AdaBoostClassifier(),
                "Linear SVM": SVC(kernel="linear", C=0.025),
                "RBF SVM": SVC(gamma=2, C=1),
                # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)), #too slow to sit through testing
                "QDA": QuadraticDiscriminantAnalysis() #warnings report a lot of colinearity. 
                                                        #I think that's okay with regard to model performance, 
                                                        # but if you're looking at the individual coefficients in the model, 
                                                        # you're going to get a bad picture from it, so bad for explaining the model.
            }

    def benchmark(self, X: List[int], Y: List[int], test_size: float=0.4, filename='bench.txt') -> None:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        with open(filename, 'w') as file:
            for model in self.models:
                #cross validation is a better indicator of how good a model is, but it takes a bit longer, so no fun for large data sets
                # crossValidation = cross_val_score(self.models[model], X, Y, cv=5) 
                start_time = process_time()
                self.models[model].fit(x_train, y_train)
                stop_time = process_time()
                
                # cv_string = f'Cross validation for model: {crossValidation}'
                elapsed = f'Time: {round((stop_time - start_time), 2)} s'
                rate = f'Rate: {round(self.models[model].score(x_test, y_test),4)}'
                
                formatted_output = f'{model}\n\t{rate}\t{elapsed}\n'
                
                file.write(formatted_output)
                print(formatted_output)
                