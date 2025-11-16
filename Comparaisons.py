import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_dt = confusion_matrix(y_test, y_pred)  


plt.figure(figsize=(22, 8))


plt.subplot(1, 3, 1)
sns.heatmap(conf_matrix_mlp, annot=True, fmt='d', cmap='Greys')
plt.title('MLP Confusion Matrix')
plt.xlabel('Prédictions')
plt.ylabel('Vérités réelles')


plt.subplot(1, 3, 2)
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues')
plt.title('K-Nearest Neighbors Confusion Matrix')
plt.xlabel('Prédictions')
plt.ylabel('Vérités réelles')


plt.subplot(1, 3, 3)
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Reds')  # Vous pouvez changer la palette de couleurs ici
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Prédictions')
plt.ylabel('Vérités réelles')


plt.tight_layout()
plt.show()
