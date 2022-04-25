from seclea_ai.cbe_sklearn.submodule.seclea_scikit_learn.sklearn import tree

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
print(clf.predict([[1, 1]]))
