# Rapport de nettoyage du projet

Date : 31 mai 2026

## Perimetre

Nettoyage organisationnel uniquement. Aucune logique metier, aucun modele, aucun poids, aucune recommandation, aucun stress test, aucun backtesting, aucun scoring, aucun Pareto, aucune frontiere efficiente et aucune allocation 10 MD n'a ete modifie.

## Fichiers et dossiers supprimes

- Caches Python et Jupyter : `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`.
- Metadonnees locales : `.idea/`, `.pytest_cache/`.
- Brouillons et backups : `_cells_dump.txt`, `sample.ipynb`, `notebooks/backups/`, `notebooks/02_optimisation_portefeuille - backup.ipynb`, `notebooks/exports/`.
- Anciens outputs dupliques : `exports/`, `outputs/`.
- Anciens exports top-level sous `data/exports/`, en conservant les dossiers finaux `notebook_01/`, `notebook_02/`, `optimization_inputs/` et le fichier `diagnostic_pre_optimisation_2025.xlsx` encore requis par les tests 10 MD.
- Caches bruts volumineux et regenerables : `data/raw/tunisie_clearing_bulletins/`, `data/raw/tunisie_yield_curve/`.
- Copie brute redondante du portefeuille : `data/raw/Maghrebia Portfolio.xlsx`.

## Fichiers conserves

- Notebooks finaux.
- Package `src/`.
- Donnees traitees et inputs d'optimisation necessaires.
- Exports finaux Notebook 01 et Notebook 02.
- Workbook `data/exports/diagnostic_pre_optimisation_2025.xlsx`, conserve comme dependance d'execution de l'analyse 10 MD.
- Scripts de controle qualite.
- Tests unitaires.
- `requirements.txt`, `.gitignore`, `README.md`, `pyproject.toml`.

## Fichiers modifies

- `README.md` : documentation complete du projet, installation, pipeline, confidentialite et commandes de reproduction.
- `requirements.txt` : dependances alignees avec les imports reels du projet.
- `.gitignore` : exclusion des caches, environnements locaux, donnees confidentielles et outputs temporaires.
- `src.zip` : regenere depuis le dossier `src/` nettoye.

## Controles effectues

- Ouverture logique du Notebook 02 par lecture `nbformat`.
- Absence d'erreur d'execution visible dans les outputs du notebook.
- Verification de la recommandation finale : `Mean_CVaR_99_5` sous `ExAnte_Central`.
- Verification de la presence des stress narratifs et du backtesting des 10 pires seances 2025.
- Verification de `Stress_Backtest_Non_Regression_Check = PASSED`.
- Verification de l'import du pipeline actif `run_notebook02_pipeline`.
- Verification de l'absence de `__pycache__` et `*.pyc`.
- Verification de `src.zip` sans cache Python.

## Avertissement

Les donnees internes Maghrebia restent confidentielles. Avant toute publication publique, verifier que les fichiers non anonymises ne sont pas ajoutes au commit Git final.
