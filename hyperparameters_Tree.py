
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


dt = DecisionTreeClassifier(random_state=42)



param_grid_dt = {
    'max_depth': [10, 20, 30, None]
    
}

# Appliquer Grid Search pour Decision Tree
grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=3, verbose=2, n_jobs=-1)
grid_search_dt.fit(X_train, y_train)


print("Meilleurs paramètres pour Decision Tree: ", grid_search_dt.best_params_)
print("Meilleure précision pour Decision Tree: ", grid_search_dt.best_score_)
