from sklearn.model_selection import GridSearchCV
from sklearn import neighbors, metrics


param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15,50,100,10000]}


score = 'accuracy'

# Créer un classifieur kNN 
clf = GridSearchCV(
    neighbors.KNeighborsClassifier(), 
    param_grid,                        
    cv=5,                              
    scoring=score                      
)


clf.fit(X_train, y_train)

print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
print(clf.best_params_)




print("\nRésultats de la validation croisée :")
print("{:<15} {:<15} {:<15}".format("n_neighbors", "Mean Accuracy", "Confidence Interval (95%)"))
print("="*45)

for mean, std, params in zip(
        clf.cv_results_['mean_test_score'],  
        clf.cv_results_['std_test_score'],   
        clf.cv_results_['params']           
    ):
    print("{:<15} {:.3f} {:<15}".format(
        params['n_neighbors'],
        mean,
        f"(± {std * 2:.3f})"  # Intervalle de confiance à 95%
    ))



 
