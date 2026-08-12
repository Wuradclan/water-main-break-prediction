# Roadmap

## Maintenant — MVP MLOps

- [x] Données brutes et snapshots
- [x] Prétraitement temporel avec protections anti-fuite
- [x] Entraînement de plusieurs familles de modèles
- [x] Tracking MLflow
- [x] Model Gate et champion
- [x] API FastAPI et interface Streamlit
- [x] Docker Compose et tests automatisés

## Prochaine itération

- [ ] Insérer les valeurs de performance du run final dans la documentation et la présentation
- [ ] Ajouter une page Streamlit d'entraînement asynchrone avec logs et statut
- [ ] Ajouter une authentification avant exposition publique de l'interface
- [ ] Ajouter des visualisations de risques et de métriques dans Streamlit
- [ ] Ajouter des alertes lorsque le champion est remplacé

## Évolution données / modèles

- [ ] Intégrer longueur de conduite après validation du contrat de données
- [ ] Enrichir avec données de sol, météo, pression et environnement
- [ ] Étudier l'explicabilité (SHAP / importance des variables)
- [ ] Évaluer les modèles sur plusieurs périodes et horizons

## Production

- [ ] Mettre en place CI/CD avec un runner auto-hébergé sur Oracle Cloud
- [ ] Mettre MLflow sur un backend et un stockage d'artefacts durables
- [ ] Ajouter un gestionnaire de tâches durable pour les entraînements longs
- [ ] Ajouter surveillance, sauvegardes et gestion des secrets
