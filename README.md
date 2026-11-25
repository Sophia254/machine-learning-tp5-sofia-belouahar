# TP5 – Sélection de variables (filtre et wrapper)

TP de Machine Learning (Master 1, ENSTA) sur le dataset Breast Cancer Wisconsin (Diagnostic) : 569 patients, 30 variables, cible = tumeur maligne (0) ou bénigne (1). Le modèle utilisé partout est une régression logistique.

Contenu du notebook :
- A : inspection des données et baseline (30 variables : 0,988 d'accuracy en test)
- B : méthodes filtre (variance, corrélation, SelectKBest avec F-test et information mutuelle)
- C : méthodes wrapper (forward, backward, RFE, RFECV) et coût de calcul
- D : ajout de 20 variables de bruit, fuite de données, comparaison finale sur le test

---

## Forward selection (partie C.1)

**Principe.** On part d'un ensemble vide et on ajoute les variables une par une. À chaque étape, on essaie d'ajouter chaque variable restante, on entraîne la régression logistique et on mesure l'accuracy en validation croisée (5 plis, sur le train). On garde la variable qui donne le meilleur score, et on recommence.

**Ce qui est fait dans le TP.**
1. Avec `SequentialFeatureSelector(direction="forward", n_features_to_select=10)` de scikit-learn : on obtient 10 variables (mean radius, mean texture, mean perimeter, mean compactness, texture error, smoothness error, worst radius, worst texture, worst perimeter, worst smoothness).
2. Avec ma propre fonction `forward_selection` (sans scikit-learn). Elle utilise deux listes, `choisies` et `restantes`, et une boucle qui teste chaque candidat avec `cross_val_score`. Elle s'arrête à 10 variables ou quand le gain de score est inférieur à ε = 0,001.
3. Je compare les deux : mon implémentation retrouve exactement les mêmes 10 variables que scikit-learn (seul l'ordre change), avec des temps proches (environ 5 s).

**Résultats.**
- Le score monte vite : 0,922 avec 1 variable (worst radius), 0,967 avec 2, 0,982 avec 3, puis il stagne autour de 0,985 et finit à 0,9899 avec 10 variables.
- Avec le critère ε = 0,001 l'algorithme s'arrête à **4 variables** (0,9849), parce que la 5e ne change pas le score. Ce critère est un peu myope : en continuant on gagne encore un peu.
- Sur le test final (D.3), le forward à 4 variables donne 0,9415 d'accuracy.
- Avec les 20 variables de bruit (D.1), le forward n'en choisit aucune sur 10.

**Coût (C.4).** Pour 10 variables sur 30 avec 5 plis : 30 + 29 + … + 21 = 255 sous-ensembles testés, soit 255 × 5 = 1275 modèles entraînés.

---

## Backward elimination (partie C.2)

**Principe.** C'est l'inverse : on part des 30 variables et on en enlève une à chaque étape. On essaie de retirer chaque variable, on mesure l'accuracy en validation croisée, et on supprime celle dont le retrait dégrade le moins le score (ou l'améliore).

**Ce qui est fait dans le TP.**
1. Avec `SequentialFeatureSelector(direction="backward", n_features_to_select=10)` : on obtient 10 variables (texture error, area error, smoothness error, fractal dimension error, worst radius, worst texture, worst perimeter, worst smoothness, worst concavity, worst symmetry).
2. Avec ma propre fonction `backward_elimination` : elle part des 30 variables et supprime une variable à chaque étape jusqu'à n'en garder qu'une, en enregistrant le score à chaque étape pour tracer la courbe « accuracy en fonction du nombre de variables restantes ». Elle retrouve les mêmes 10 variables que scikit-learn.
3. Je compare avec le forward.

**Résultats.**
- L'accuracy CV passe de 0,9824 avec les 30 variables à 0,9899 vers 15–21 variables, donc enlever des variables redondantes aide un peu. Avec 10 variables on est à 0,9849, puis le score baisse quand il en reste très peu.
- Forward et backward n'ont que **6 variables en commun** sur 10 (smoothness error, texture error, worst perimeter, worst radius, worst smoothness, worst texture).
- Sur le test final (D.3), le backward à 10 variables donne 0,9591 d'accuracy.
- Avec les 20 variables de bruit (D.1), le backward en garde **5 sur 10** : c'est la méthode la moins robuste au bruit.

**Coût (C.2 et C.4).** Pour passer de 30 à 10 variables : 30 + 29 + … + 11 = 410 sous-ensembles testés, soit 410 × 5 = 2050 modèles. C'est pour cela qu'il est plus lent que le forward (environ 9 s contre 2 s en D.3).

---

## Pourquoi forward et backward ne donnent pas la même chose

Les deux méthodes sont gloutonnes : elles prennent la meilleure décision du moment et ne reviennent jamais dessus. Le forward cherche la variable qui aide le plus (au début, avec très peu de variables dans le modèle), alors que le backward cherche celle dont on peut se passer le plus facilement (en présence de toutes les autres). Comme beaucoup de variables sont presque identiques (radius, perimeter, area), plusieurs choix donnent un score presque égal, et un tout petit écart suffit à changer le résultat. Aucune des deux ne garantit donc le sous-ensemble optimal : ici leurs ensembles de 10 variables sont différents et ont des scores différents (0,9899 pour le forward, 0,9849 pour le backward).