from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier(random_state=42)


param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)]
}


grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, cv=3, verbose=2, n_jobs=-1)
grid_search_mlp.fit(X_train, y_train)


print("Meilleurs paramètres pour MLP: ", grid_search_mlp.best_params_)
print("Meilleure précision pour MLP: ", grid_search_mlp.best_score_)
