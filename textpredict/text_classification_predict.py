#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Nguyễn Thành Thủy, email: thuynt@due.edu.vn
# Trường Đại học Kinh tế, Đại học Đà Nẵng.
# Dự án Chatbot VIETNAM-AIRLINE-Assistant

from common import file
from model.LogisticRegression_Model import LogisticRegression_Model
from model.NaiveBayes_Model import NaiveBayes_Model
from model.SVM_Model import SVM_Model
from model.KNeighborsClassifier_Model import KNeighborsClassifier_Model
from model.DecisionTreeClassifier_Model import DecisionTreeClassifier_Model

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class TextClassificationPredict(object):
    def __init__(self, question_test, db_train, db_train_extend, db_answers):
        self.select_model(1)
        self.question_test = question_test
        self.db_train = db_train
        self.db_answers = db_answers
        self.db_train_extend = db_train_extend

    def select_model(self, i_model):
        if i_model == 1:
            self.model = SVM_Model()
        elif i_model == 2:
            self.model = LogisticRegression_Model()
        elif i_model == 3:
            self.model = NaiveBayes_Model()
        elif i_model == 4:
            self.model = KNeighborsClassifier_Model()


    def Text_Predict(self):
        fallback_intent = file.get_fallback_intent()

        df_train_extend = pd.DataFrame(self.db_train_extend)
        df_train = pd.DataFrame(self.db_train)
        df_answers = pd.DataFrame(self.db_answers)  # Tập câu trả lời

        db_Predict = []
        db_Predict.append({"Question": self.question_test})
        df_Predict = pd.DataFrame(db_Predict)

        # Predict in Question Text
        for i in [1, 2, 3, 4]:
            self.select_model(i)
            clf = self.model.clf.fit(df_train_extend["Question"], df_train_extend.Intent)
            list_score = clf.predict_proba(df_Predict["Question"]).flatten()  # --> array of score
            predicted = list_score.tolist().index(list_score.max())

            if (list_score[predicted] >= 0.5):
                break

        if (list_score[predicted] >= 0.5):
            print("Model: ", self.model.__class__.__name__)
            mess = "Hỏi: " + self.question_test + "\n"
            mess += "Trả lời: "
            mess += df_train["Question"][predicted].upper() \
                    + "\n" + df_answers["Answers"][predicted] \
                    + "\n(Score:" + str(round(list_score[predicted], 3)) \
                    + " | Intent:" + str(predicted+1) + ")"
            print(mess)

        else:
            print("Hỏi: " + self.question_test + "\n")
            mess = fallback_intent[random.randint(0, len(fallback_intent)-1)]
            print(mess)

        return mess

    def GridSearchCV(self):
        df_train_extend = pd.DataFrame(self.db_train_extend)
        X = df_train_extend.Question
        y = df_train_extend.Intent

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=0)

        if (self.model.__class__.__name__ == "LogisticRegression_Model"):
            C = np.logspace(-6, 6, 50)
            max_iter = ([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300])
            parameters = dict(clf__C = C, clf__max_iter = max_iter)

            cv = GridSearchCV(self.model.clf, parameters)
            t0 = time()
            cv.fit(X_train, y_train)
            print("Done in {0}s".format(time() - t0))
            print('Best C: %.2f' % cv.best_estimator_.get_params()['clf__C'])
            print('Best max_iter:', cv.best_estimator_.get_params()['clf__max_iter'])

            y_train_pred = classification_report(y_train, cv.predict(X_train))
            y_test_pred = classification_report(y_test, cv.predict(X_test))

            print("""{model_name}\n Train Accuracy: \n{train} 
                \n Test Accuracy:  \n{test}""".format(model_name=self.model.__class__.__name__,
                                                      train=y_train_pred, test=y_test_pred))

        elif (self.model.__class__.__name__ == "NaiveBayes_Model"):
            parameters = {
                'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
            }


            cv = GridSearchCV(self.model.clf, parameters)
            t0 = time()
            cv.fit(X_train, y_train)
            print("Done in {0}s".format(time() - t0))
            print('Best alpha: %.2f' % cv.best_estimator_.get_params()['clf__alpha'])

            y_train_pred = classification_report(y_train, cv.predict(X_train))
            y_test_pred = classification_report(y_test, cv.predict(X_test))

            print("""{model_name}\n Train Accuracy: \n{train} 
                \n Test Accuracy:  \n{test}""".format(model_name=self.model.__class__.__name__,
                                                      train=y_train_pred, test=y_test_pred))

        elif (self.model.__class__.__name__ == "SVM_Model"):
            parameters = {'clf__C': [1e3, 5e3, 1e4, 5e4, 1e5],
                          'clf__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

            cv = GridSearchCV(self.model.clf, parameters)
            t0 = time()
            cv.fit(X_train, y_train)

            print("Done in {0}s".format(time() - t0))
            print('Best C: %.2f' % cv.best_estimator_.get_params()['clf__C'])
            print('Best gamma: ', cv.best_estimator_.get_params()['clf__gamma'])

        elif (self.model.__class__.__name__ == "DecisionTreeClassifier_Model"):
            parameters = {"clf__max_depth": range(1, 100)}

            cv = GridSearchCV(self.model.clf, parameters)
            t0 = time()
            cv.fit(X_train, y_train)

            print("Done in {0}s".format(time() - t0))
            print('Best max_depth: %.2f' % cv.best_estimator_.get_params()['clf__max_depth'])

            y_train_pred = classification_report(y_train, cv.predict(X_train))
            y_test_pred = classification_report(y_test, cv.predict(X_test))

            print("""{model_name}\n Train Accuracy: \n{train} 
                \n Test Accuracy:  \n{test}""".format(model_name=self.model.__class__.__name__,
                                                      train=y_train_pred, test=y_test_pred))

        elif (self.model.__class__.__name__ == "KNeighborsClassifier_Model"):
            parameters = {"clf__n_neighbors": range(1, 100)}

            cv = GridSearchCV(self.model.clf, parameters)
            t0 = time()
            cv.fit(X_train, y_train)

            print("Done in {0}s".format(time() - t0))
            print('Best n_neighbors: %.2f' % cv.best_estimator_.get_params()['clf__n_neighbors'])

            y_train_pred = classification_report(y_train, cv.predict(X_train))
            y_test_pred = classification_report(y_test, cv.predict(X_test))

            print("""{model_name}\n Train Accuracy: \n{train} 
                \n Test Accuracy:  \n{test}""".format(model_name=self.model.__class__.__name__,
                                                      train=y_train_pred, test=y_test_pred))


    def Test_Model_ByNormal(self):
        df_train_extend = pd.DataFrame(self.db_train_extend)
        X = df_train_extend.Question
        y = df_train_extend.Intent

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=1)

        self.model.clf.fit(X_train, y_train)  # Training Model

        y_train_pred = classification_report(y_train, self.model.clf.predict(X_train))
        y_test_pred = classification_report(y_test, self.model.clf.predict(X_test))

        print("""{model_name}\n Train Accuracy: \n{train} 
        \n Test Accuracy:  \n{test}""".format(model_name=self.model.__class__.__name__,
                                              train=y_train_pred, test=y_test_pred))



        #Train Accuracy: precision    recall    f1-score    support
        #avg / total       0.99         0.99        0.99         160
        #Test Accuracy:
        #avg / total       0.87         0.76        0.77         41

        #predictions = cross_val_predict(self.model.clf, X, y)
        #plt.scatter(y, predictions)
        #plt.show()

        #Plotting Learning Curves
        #title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        #self.plot_learning_curve(self.model.clf, title, X, y)
        #plt.show()


    def Test_Model_ByKFold(self):
        df_train_extend = pd.DataFrame(self.db_train_extend)
        #df_train = pd.DataFrame(self.db_train)

        X = df_train_extend.Question
        y = df_train_extend.Intent

        kf = KFold(n_splits=len(df_train_extend), random_state=None, shuffle=False)
        kf.get_n_splits(X)

        _precision_score = 0
        _recall_score = 0
        _accuracy = 0
        count = 0

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.clf.fit(X_train, y_train)  # Training Model
            print(test_index)

            y_train_pred = cross_val_predict(self.model.clf, X_train, y_train)
            _precision_score = _precision_score + precision_score(y_train, y_train_pred, average='weighted')
            _recall_score = _recall_score + recall_score(y_train, y_train_pred, average='weighted')

            pred = self.model.clf.predict(X_test)
            _accuracy += accuracy_score(y_test, pred)

            count = count + 1

        MacroAVG_Precision_Score = _precision_score / count
        MacroAVG_Recall_Score = _recall_score/count
        MacroAVG_FScore = 2*MacroAVG_Precision_Score*MacroAVG_Recall_Score/(MacroAVG_Precision_Score+MacroAVG_Recall_Score)
        print("Macro-average Precision-Score: ", MacroAVG_Precision_Score)
        print("Macro-average Recall-Score: ", MacroAVG_Recall_Score)
        print("Macro-average F-Score: ", MacroAVG_FScore)
        print("accuracy: ", _accuracy/count)


        #OK: Compute confusion matrix
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        y_pred = self.model.clf.fit(X_train, y_train).predict(X_test)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=1)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.model.plot_confusion_matrix(cnf_matrix, classes=df_train.Intent,
                              title='Confusion matrix, without normalization')
        plt.show()
        """
        # -----------------

        #OK: Plotting Learning Curves
        """
        title = "Learning Curves (Test model by Leave-One-Out Method)"
        self.model.plot_learning_curve(self.model.clf, title, X, y, cv = kf)
        plt.show()
        """
        #--


        #OK: Make cross validated predictions
        """
        cv=KFold(n_splits=len(df_train_extend), random_state=None, shuffle=False)
        predictions = cross_val_predict(self.model.clf, X, y, cv=cv)
        plt.scatter(y, predictions)
        plt.show()
        """
        #---


    def Test_Model_ByLeaveOneOut(self):
        df_train_extend = pd.DataFrame(self.db_train_extend)
        X = df_train_extend.Question
        y = df_train_extend.Intent

        clf = self.model.clf
        cv = LeaveOneOut()
        cv.get_n_splits(X)

        _precision_score = 0
        _recall_score = 0
        _accuracy = 0
        count = 0

        for train_index, test_index in cv.split(X):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            clf.fit(X_train, y_train)

            y_train_pred = cross_val_predict(self.model.clf, X_train, y_train)
            _precision_score = _precision_score + precision_score(y_train, y_train_pred, average='weighted')
            _recall_score = _recall_score + recall_score(y_train, y_train_pred, average='weighted')

            pred = self.model.clf.predict(X_test)
            _accuracy += accuracy_score(y_test, pred)

            count = count + 1
            print(count)

            #OK--classification_report
            """
            y_train_pred = classification_report(y_train, self.model.clf.predict(X_train))
            y_test_pred = classification_report(y_test, self.model.clf.predict(X_test))
            print("Train Accuracy:\n", y_train_pred,
                "Test Accuracy:\n",y_test_pred)
            """
            #--



        #OK - Classification_report
        MacroAVG_Precision_Score = _precision_score / count
        MacroAVG_Recall_Score = _recall_score/count
        MacroAVG_FScore = 2*MacroAVG_Precision_Score*MacroAVG_Recall_Score/(MacroAVG_Precision_Score+MacroAVG_Recall_Score)
        print("Macro-average Precision-Score: ", MacroAVG_Precision_Score)
        print("Macro-average Recall-Score: ", MacroAVG_Recall_Score)
        print("Macro-average F-Score: ", MacroAVG_FScore)
        print("Accuracy: ", _accuracy/count)
        #---



        #OK - Plotting Learning Curves
        """
        print("Plotting Learning Curves, Waiting...")
        title = "Learning Curves (Test model by Leave-One-Out Method)"
        self.model.plot_learning_curve(clf, title, X, y, cv=cv)
        plt.show()
        """
        #---


        #OK - Make cross validated predictions
        """
        print("Make cross validated predictions, Waiting...")
        cv = LeaveOneOut()
        predictions = cross_val_predict(clf, X, y, cv=cv)

        fig, ax = plt.subplots()
        ax.scatter(y, predictions, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
        """
        #---


        #OK: Compute confusion matrix
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        y_pred = self.clf.fit(X_train, y_train).predict(X_test)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=1)

        # Plot non-normalized confusion matrix
        plt.figure()

        df_train = pd.DataFrame(self.db_train)
        self.model.plot_confusion_matrix(cnf_matrix, classes=df_train.Intent,
                              title='Confusion matrix, without normalization')
        plt.show()
        """
        # -----------------


if __name__ == '__main__':

    #txt = u"Làm gì để đảm bảo các điều kiện theo quy định của vietnam airlines"
    #txt = u"Kiện hàng chuyển phát nhanh cao bao nhiêu?"
    txt = u"Điện thoại có gửi theo đường hàng hóa không"
    #txt = u"Tôi mang hàng nguy hiểm lên máy bay"
    #txt = u"có kho mát không?"
    #txt = u"Tôi mang hàng nguy hiểm lên máy bay có được không"

    predict = TextClassificationPredict(txt, file.get_dbtrain(), file.get_dbtrain_extend(), file.get_dbanswers())

    #predict.GridSearchCV()
    #predict.Test_Model_ByLeaveOneOut()
    predict.Text_Predict()



