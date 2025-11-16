from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


clf = DecisionTreeClassifier(max_depth= 10, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred,zero_division=1)


print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

clf = tree.DecisionTreeClassifier(max_depth= 5, random_state=42);
clf = clf.fit(X_test, y_pred)
plt.figure(figsize=(150,100))
tree.plot_tree(clf,fontsize=30);
