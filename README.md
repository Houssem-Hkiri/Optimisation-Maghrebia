# PFE Maghrebia - Optimisation du portefeuille d'actifs financiers

Projet de fin d'etudes en finance quantitative consacre a l'optimisation du portefeuille d'actifs financiers d'une compagnie d'assurances, applique au cas de Maghrebia Assurances.

Le projet construit une chaine reproductible allant des donnees de marche et du portefeuille actuel jusqu'a une recommandation d'allocation sous contraintes, avec diagnostic rendement-risque, rendements attendus hybrides ex ante, covariance robuste, optimisation, stress tests, backtesting et allocation additionnelle de 10 MD.

## Objectif du PFE

L'objectif est de fournir un cadre d'aide a la decision permettant de :

- nettoyer et harmoniser les donnees financieres disponibles ;
- valoriser les actifs et construire des rendements exploitables ;
- estimer des rendements attendus ex ante selon une approche hybride ;
- stabiliser le risque via une covariance Ledoit-Wolf ;
- comparer plusieurs familles d'optimisation de portefeuille ;
- verifier les contraintes internes et reglementaires applicables ;
- analyser la robustesse par stress tests et backtesting ;
- produire une recommandation d'allocation interpretable pour la direction ;
- preparer des exports reutilisables par un dashboard decisionnel.

Le notebook principal d'optimisation ne cherche pas a produire une prevision certaine. Les resultats sont presentes comme des portefeuilles recommandes sous hypotheses, donnees disponibles et contraintes specifiees.

## Structure du dossier

```text
.
├── notebooks/
│   ├── 00_extract_daily_yield_curves_2025.ipynb
│   ├── 00_extract_tunisie_clearing_corporate_spreads.ipynb
│   ├── 01_diagnostic_pre_optimisation.ipynb
│   └── 02_optimisation_portefeuille.ipynb
├── src/
│   ├── data/
│   │   ├── tunisie_clearing/
│   │   └── tunisie_yield_curve/
│   └── maghrebia_quant/
│       ├── optimization/
│       ├── allocation_10md.py
│       ├── apt_weekly.py
│       ├── portfolio.py
│       └── risk_metrics.py
├── data/
│   ├── processed/
│   ├── exports/
│   │   ├── notebook_01/
│   │   ├── notebook_02/
│   │   └── optimization_inputs/
│   └── raw/                  # donnees confidentielles ou regenerables, ignorees par Git
├── scripts/
│   └── quality_check_notebook_02.py
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Notebooks

### 00 - Extraction des courbes

Les notebooks `00_extract_daily_yield_curves_2025.ipynb` et `00_extract_tunisie_clearing_corporate_spreads.ipynb` documentent l'extraction ou la preparation des courbes souveraines et des spreads corporate.

Ces notebooks sont utiles pour la tracabilite des donnees de marche. Les caches de collecte, les PDF bruts et les fichiers temporaires ne sont pas necessaires au rendu final.

### 01 - Diagnostic pre-optimisation

`notebooks/01_diagnostic_pre_optimisation.ipynb` construit les inputs robustes utilises par l'optimisation :

- portefeuille actuel et poche optimisable ;
- rendements historiques ;
- valorisation des actifs obligataires ;
- rendements attendus hybrides ex ante ;
- scenarios `ExAnte_Prudent`, `ExAnte_Central`, `ExAnte_Optimistic` ;
- scenario `Historical_Raw_Comparative` uniquement descriptif ;
- covariance Ledoit-Wolf ;
- diagnostics PCA ;
- metriques de risque ;
- quality flags ;
- fichiers intermediaires pour le Notebook 02.

Les outputs principaux sont dans :

```text
data/exports/notebook_01/
data/exports/optimization_inputs/
data/processed/
```

### 02 - Optimisation et allocation

`notebooks/02_optimisation_portefeuille.ipynb` est le livrable principal d'aide a la decision.

Il appelle le pipeline actif :

```python
from maghrebia_quant.optimization import run_notebook02_pipeline
```

Le notebook orchestre les calculs du package sans logique parallele. Il affiche les resultats sous forme de tableaux et graphiques Plotly, puis regenere les exports finaux.

Elements attendus dans la version finale :

- recommandation centrale : `Mean_CVaR_99_5` sous `ExAnte_Central` ;
- affichage en francais professionnel : scenario central de rendement attendu, recommandation centrale, donnees critiques manquantes, valide avec reserves ;
- stress tests narratifs ;
- backtesting des 10 pires seances 2025 ;
- frontiere efficiente ;
- Monte Carlo exploratoire ;
- Pareto filter ;
- scoring multicritere ;
- allocation additionnelle de 10 MD ;
- synthese executive.

Les exports principaux sont dans :

```text
data/exports/notebook_02/02_optimisation_outputs.xlsx
data/exports/notebook_02/02_optimisation_portefeuille.html
data/exports/notebook_02/figures/
```

## Installation

### 1. Creer un environnement Python

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name pfe-maghrebia --display-name "Python (PFE Maghrebia)"
```

### 2. Lancer Jupyter

```powershell
jupyter lab
```

Ouvrir ensuite les notebooks dans l'ordre :

```text
notebooks/00_extract_daily_yield_curves_2025.ipynb
notebooks/00_extract_tunisie_clearing_corporate_spreads.ipynb
notebooks/01_diagnostic_pre_optimisation.ipynb
notebooks/02_optimisation_portefeuille.ipynb
```

Pour relancer uniquement le Notebook 02 en ligne de commande :

```powershell
python -m jupyter nbconvert --to notebook --execute notebooks/02_optimisation_portefeuille.ipynb --inplace --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=python3
python -m jupyter nbconvert --to html notebooks/02_optimisation_portefeuille.ipynb --output 02_optimisation_portefeuille --output-dir data/exports/notebook_02 --HTMLExporter.embed_images=True
```

## Description courte du pipeline

1. Les donnees du portefeuille, les prix de marche, les courbes zero-coupon et les spreads corporate sont charges et controles.
2. Le Notebook 01 construit les rendements historiques, les rendements attendus hybrides ex ante, la covariance Ledoit-Wolf et les diagnostics de risque.
3. Le Notebook 02 charge ces outputs sans reconstruire les rendements attendus.
4. Les portefeuilles candidats sont generes par plusieurs familles de modeles : portefeuille actuel, minimum variance, Markowitz, Mean-CVaR, Risk Parity, Robust-CVaR et benchmarks comparatifs.
5. Monte Carlo est utilise comme exploration de l'espace faisable, pas comme frontiere efficiente exacte.
6. Chaque portefeuille est evalue avec les memes metriques : rendement attendu, volatilite, VaR, CVaR, stress loss, HHI, distance au portefeuille actuel, contraintes et score multicritere.
7. Le Pareto filter et le scoring multicritere selectionnent une recommandation centrale et des alternatives prudentes.
8. L'allocation additionnelle de 10 MD est analysee separement et ne doit pas etre interpretee comme garantie mecanique du ROE global comptable.

## Confidentialite des donnees Maghrebia

Les fichiers de donnees internes de Maghrebia Assurances ne doivent pas etre publies dans un depot GitHub public.

En particulier, les fichiers suivants sont consideres confidentiels ou sensibles :

- portefeuille Maghrebia nominatif ;
- fichiers Excel internes ;
- donnees brutes placees dans `data/raw/` ;
- documents sources non anonymises ;
- exports contenant des informations non publiques.

Le fichier `.gitignore` exclut les donnees brutes, les fichiers temporaires et les documents confidentiels usuels. Pour une diffusion publique, remplacer les donnees internes par des exemples anonymises ou fournir uniquement les outputs valides autorises par l'entreprise.

## Exports finaux a transmettre

Pour un envoi aux encadrants, les fichiers les plus importants sont :

```text
notebooks/01_diagnostic_pre_optimisation.ipynb
notebooks/02_optimisation_portefeuille.ipynb
data/exports/notebook_01/01_diagnostic_outputs.xlsx
data/exports/notebook_01/01_diagnostic_pre_optimisation.html
data/exports/notebook_02/02_optimisation_outputs.xlsx
data/exports/notebook_02/02_optimisation_portefeuille.html
data/exports/notebook_02/figures/
src/
requirements.txt
README.md
```

## Tests et controles

Tests unitaires :

```powershell
pytest -q
```

Controle qualite du Notebook 02 :

```powershell
python scripts/quality_check_notebook_02.py
```

Verification rapide de la recommandation finale :

```powershell
python - <<'PY'
import pandas as pd
path = "data/exports/notebook_02/02_optimisation_outputs.xlsx"
rec = pd.read_excel(path, sheet_name="Final_Recommendation")
print(rec[["Model", "Scenario"]].head(1))
PY
```

La version valide attendue affiche :

```text
Model    Mean_CVaR_99_5
Scenario ExAnte_Central
```

## Limites methodologiques documentees

- Les rendements attendus ex ante dependent des hypotheses construites dans le Notebook 01.
- La CVaR 99,5 % est un indicateur interne prudentiel de risque extreme, pas un SCR reglementaire complet.
- Les stress tests dependants de donnees de duration ou de spreads peuvent etre marques comme donnees critiques manquantes.
- Certaines contraintes reglementaires restent non testables si les donnees externes necessaires, comme le capital social ou des donnees detaillees par emetteur, ne sont pas disponibles.
- Les resultats ne doivent pas etre interpretes comme une prevision certaine du rendement futur.

## Licence et usage

Ce projet est un support academique et professionnel pour un PFE d'ingenieur en finance quantitative. Toute diffusion externe doit respecter les regles de confidentialite de Maghrebia Assurances et les autorisations des encadrants.
