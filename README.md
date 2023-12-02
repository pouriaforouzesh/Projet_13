# Résumé du Projet

## Mission

L'objectif principal de ce projet est de construire un modèle de scoring automatique pour prédire la probabilité de faillite d'un client. En plus de cela, l'équipe doit développer un dashboard interactif destiné aux gestionnaires de la relation client pour interpréter les prédictions du modèle et améliorer leur connaissance client. Le déploiement en production du modèle et du dashboard sera effectué à l'aide d'une API sur une plateforme Cloud gratuite.

## Approche

Le manager, Michaël, suggère d'utiliser des kernels Kaggle pour faciliter l'analyse exploratoire, la préparation des données, et le feature engineering. L'utilisation de Dash, Bokeh, ou Streamlit est recommandée pour le dashboard interactif. De plus, une démarche MLOps sera mise en place pour automatiser le cycle de vie du modèle, en utilisant des outils open source.

## Spécifications Techniques

1. **Dashboard :** Le dashboard interactif doit permettre de visualiser le score et son interprétation pour chaque client de manière compréhensible. Il devrait également offrir des fonctionnalités de filtrage, de comparaison des clients, et d'affichage d'informations descriptives.

2. **Technologies :** Michaël propose l'utilisation de Dash, Bokeh, ou Streamlit pour le dashboard, et une démarche MLOps avec des outils open source pour la gestion du cycle de vie du modèle.

3. **Data Drift :** L'outil evidently sera utilisé pour détecter le Data Drift entre les datas d'entraînement et les datas de production, en supposant que le dataset "application_train" représente les données de modélisation et le dataset "application_test" représente les données de nouveaux clients en production.

## Spécifications Contextuelles

1. **Déséquilibre des Classes :** Le déséquilibre entre les bons et mauvais clients doit être pris en compte. Une méthode au choix doit être utilisée pour élaborer un modèle pertinent.

2. **Coût Métier :** Le coût métier entre un faux négatif et un faux positif doit être pris en compte. Un score métier sera créé pour comparer les modèles en optimisant le seuil de classification.

## Livrables

1. **Déploiement :** L'application dashboard et l'API seront déployées sur une plateforme Cloud gratuite.

2. **Dossier de Code :** Un dossier géré via un outil de versioning contiendra le code de modélisation, le code du dashboard, et le code pour déployer le modèle en API.

3. **Analyse Data Drift :** Un tableau HTML d'analyse de Data Drift utilisant evidently sera fourni.

4. **Note Méthodologique :** Une note technique détaillera la méthodologie d'entraînement, le traitement du déséquilibre des classes, la fonction coût métier, les résultats, l'interprétabilité du modèle, les limites et les améliorations possibles.

5. **Présentation :** Un support de présentation pour la soutenance, détaillant le travail réalisé, les commits, le dossier Github, les tests unitaires, et le déploiement continu.
