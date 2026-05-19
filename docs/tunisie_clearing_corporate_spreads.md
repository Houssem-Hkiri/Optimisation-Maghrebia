# Pipeline Tunisie Clearing - spreads corporate 2023-2025

Ce pipeline extrait les bulletins historiques Tunisie Clearing, telecharge les PDF disponibles, extrait le texte et isole les primes corporate publiees. Il ne reconstruit pas de spread absent et ne complete pas automatiquement les valeurs manquantes.

## Methode

1. `scrape_bulletins.py` lit la page historique avec `requests` et `BeautifulSoup`. Si les PDF ne sont pas visibles dans le HTML statique, il interroge l'API publique Angular de Tunisie Clearing, puis bascule vers Playwright headless uniquement si ces deux approches echouent.
2. `download_bulletins.py` telecharge les PDF dans `data/raw/tunisie_clearing_bulletins/YYYY/` et verifie la signature `%PDF-`.
3. `extract_pdf_text.py` extrait le texte avec `pdfplumber`, puis `pypdf` en secours. Les textes complets sont conserves dans `data/interim/tunisie_clearing/text/`.
4. `extract_corporate_spreads.py` cherche les sections corporate, les secteurs normalises et les formulations de prime directe ou de passage de l'ancienne prime a la nouvelle prime.
5. `build_corporate_spreads_dataset.py` normalise les pourcentages, exporte le CSV/XLSX final et produit un rapport qualite Excel multi-onglets.

## Limites finance

- 2023 : la courbe corporate officielle est signalee comme probablement absente avant fin decembre 2023. Le pipeline produit un flag dedie si aucun spread officiel n'est detecte.
- 2024 : les formulations peuvent etre narratives et centrees sur le leasing; le pipeline extrait l'ancienne prime et la prime finale lorsque la phrase contient les deux valeurs.
- 2025 : la couverture attendue est plus large, notamment bancaire, leasing et microfinance, mais chaque observation doit rester reliee a un extrait source.

## Execution

Depuis la racine du projet :

```powershell
$env:PYTHONPATH="src"
python -m data.tunisie_clearing.build_corporate_spreads_dataset
```

Sorties principales :

- `data/interim/tunisie_clearing/bulletins_index.csv`
- `data/interim/tunisie_clearing/extracted_text_index.csv`
- `data/interim/tunisie_clearing/corporate_spreads_raw.csv`
- `data/processed/tunisie_clearing/corporate_spreads_2023_2025_clean.csv`
- `data/processed/tunisie_clearing/corporate_spreads_2023_2025_clean.xlsx`
- `data/processed/tunisie_clearing/corporate_spreads_quality_report.xlsx`
