from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn,zero_division=1)





print(f"Accuracy of KNN: {accuracy_knn * 100:.2f}%")
print("\nConfusion Matrix for KNN:")
print(conf_matrix_knn)
print("\nClassification Report for KNN:")
print(class_report_knn)
