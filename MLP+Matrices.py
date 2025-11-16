from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA




'''

j'éssayais de réduire en dimension afin d'exploiter l'algorithme. 
Car je pensais qu'au dela de 99% on étatit forcément dans une situation d'overfiting. Mais ce n'est pas forcément le cas. 
J'ai compris cela après une longue discution avec le professeur. 
J'ai jugé pertinent de laisser cette partie en commentaire car elle témoigne de l'évolution de mon travail et de ma compréhension.


pca = PCA(n_components=0.95)   # Garde 95% de la variance  
X_train_pca = pca.fit_transform(X_train)                       
X_test_pca = pca.transform(X_test)                             
'''

X_train_pca = X_train
X_test_pca = X_test

mlp = MLPClassifier(hidden_layer_sizes=(25, 100), max_iter=400, random_state=42)
mlp.fit(X_train_pca, y_train)

y_pred_mlp = mlp.predict(X_test_pca)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
class_report_mlp = classification_report(y_test, y_pred_mlp,zero_division=1)

print(f"Accuracy of MLP: {accuracy_mlp * 100:.2f}%")
print("\nConfusion Matrix for MLP:")
print(conf_matrix_mlp)
print("\nClassification Report for MLP:")
print(class_report_mlp)
