import sklearn
from sklearn import datasets
from sklearn import svm
'''from sklearn import matrics'''
from sklearn.neighbors import KNeighborsClassifier

data=datasets.load_breast_cancer()
"""print(data.feature_names)
print(data.target_names)"""
ori=['malignant','benign']
x=data.data
y=data.target
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)
clf=svm.SVC(kernel="linear")
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
acc=clf.score(x_test,y_test)
print (acc)
