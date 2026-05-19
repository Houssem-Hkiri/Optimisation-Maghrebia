"""Génération du notebook 03 - allocation supplémentaire 10 MD multi-scénarios APT."""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def main() -> Path:
    cells = []

    cells.append(new_markdown_cell(
        "# Allocation supplémentaire de 10 MD\n\n"
        "Ce notebook analyse l'impact d'une **enveloppe additionnelle de 10 MD** investie **uniquement "
        "dans la poche optimisable** du portefeuille Maghrebia, en testant l'objectif "
        "**ROE cible = TSR + 4 %** sur les trois scénarios APT (Prudent, Central, Optimiste).\n\n"
        "Le notebook s'appuie sur les sorties méthodologiques validées des notebooks 01 (diagnostic, "
        "APT, scénarios) et 02 (optimisation). Max Sharpe est volontairement exclu, conformément aux "
        "choix méthodologiques du notebook 02."
    ))

    cells.append(new_markdown_cell(
        "## Limite méthodologique : ROE proxy\n\n"
        "L'objectif `ROE = TSR + 4 %` est interprété ici comme une **cible de rendement financier "
        "attendu**. Cette approximation ne remplace pas le ROE comptable, qui dépend également du "
        "résultat technique, des charges, de la réassurance, de la fiscalité, des provisions et "
        "des fonds propres. Le ROE comptable réel est :\n\n"
        "$$ROE = \\frac{Resultat\\ net}{Fonds\\ propres}$$\n\n"
        "Dans ce notebook, faute d'un modèle complet du compte de résultat et du bilan, on utilise "
        "comme proxy le rendement financier attendu du portefeuille. L'hypothèse TSR est lue depuis "
        "le notebook 01 et doit rester synchronisée avec le notebook 02."
    ))

    cells.append(new_code_cell(
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n"
        "from IPython.display import Markdown, display\n"
        "\n"
        "PROJECT_ROOT = Path.cwd()\n"
        "if PROJECT_ROOT.name == \"notebooks\":\n"
        "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
        "if str(PROJECT_ROOT / \"src\") not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT / \"src\"))\n"
        "\n"
        "from maghrebia_quant.allocation_10md import (\n"
        "    AdditionalAllocationConfig,\n"
        "    SCENARIO_ORDER,\n"
        "    export_multi_scenario_analysis,\n"
        "    run_multi_scenario_allocation,\n"
        ")\n"
        "\n"
        "ADDITIONAL_BUDGET = 10_000_000\n"
        "ROE_SPREAD_TARGET = 0.04\n"
        "\n"
        "CONFIG = AdditionalAllocationConfig(\n"
        "    additional_budget=ADDITIONAL_BUDGET,\n"
        "    roe_spread_target=ROE_SPREAD_TARGET,\n"
        "    monte_carlo_portfolios=20_000,\n"
        ")\n"
        "\n"
        "EXPORT_DIR = PROJECT_ROOT / \"exports\"\n"
        "FIGURE_DIR = EXPORT_DIR / \"figures\" / \"notebook_03\"\n"
        "EXCEL_PATH = EXPORT_DIR / \"notebook_03_allocation_10MD_results.xlsx\"\n"
        "FIGURE_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "pd.options.display.float_format = \"{:,.6f}\".format\n"
        "pd.options.display.max_columns = 60"
    ))

    cells.append(new_code_cell(
        "result = run_multi_scenario_allocation(PROJECT_ROOT, CONFIG)\n"
        "\n"
        "TSR = result[\"tsr\"]\n"
        "TARGET_RETURN = result[\"by_scenario\"][\"APT_Central\"][\"state\"][\"TARGET_RETURN\"]\n"
        "V_TOTAL_CURRENT = result[\"V_TOTAL_CURRENT\"]\n"
        "V_OPT_CURRENT = result[\"V_OPT_CURRENT\"]\n"
        "V_FIXED_CURRENT = result[\"V_FIXED_CURRENT\"]\n"
        "V_OPT_FINAL = V_OPT_CURRENT + ADDITIONAL_BUDGET\n"
        "V_TOTAL_FINAL = V_TOTAL_CURRENT + ADDITIONAL_BUDGET\n"
        "\n"
        "assert abs(V_TOTAL_CURRENT - (V_OPT_CURRENT + V_FIXED_CURRENT)) <= 1e-2\n"
        "assert abs(V_OPT_FINAL - (V_OPT_CURRENT + ADDITIONAL_BUDGET)) <= 1e-8\n"
        "assert abs(V_TOTAL_FINAL - (V_TOTAL_CURRENT + ADDITIONAL_BUDGET)) <= 1e-8\n"
        "\n"
        "display(Markdown(\n"
        "    f\"**TSR retenu (depuis notebook 01) :** {TSR:.2%}  \\n\"\n"
        "    f\"**TARGET_RETURN = TSR + 4 % :** {TARGET_RETURN:.2%}  \\n\"\n"
        "    f\"**V_TOTAL_CURRENT :** {V_TOTAL_CURRENT:,.0f} TND  \\n\"\n"
        "    f\"**V_OPT_CURRENT :** {V_OPT_CURRENT:,.0f} TND  \\n\"\n"
        "    f\"**V_FIXED_CURRENT :** {V_FIXED_CURRENT:,.0f} TND  \\n\"\n"
        "    f\"**ADDITIONAL_BUDGET :** {ADDITIONAL_BUDGET:,.0f} TND  \\n\"\n"
        "    f\"**V_OPT_FINAL :** {V_OPT_FINAL:,.0f} TND  \\n\"\n"
        "    f\"**V_TOTAL_FINAL :** {V_TOTAL_FINAL:,.0f} TND\"\n"
        "))"
    ))

    cells.append(new_markdown_cell(
        "## 1. Cadrage méthodologique\n\n"
        "Les 10 MD ne sont **pas** ajoutés au portefeuille global comme univers d'allocation. Ils sont "
        "ajoutés **uniquement à la poche optimisable**, qui comprend :\n\n"
        "- titres de l'État ;\n"
        "- emprunts obligataires / obligations corporate ;\n"
        "- actions cotées.\n\n"
        "Les actifs non optimisables restent figés : immobilier, OPCVM non traités, SICAR/SICAF, "
        "actions non cotées, participations, placements non modélisés."
    ))

    cells.append(new_code_cell(
        "display(result[\"01_Hypotheses\"])\n"
        "display(result[\"02_Current_Portfolio\"])"
    ))

    cells.append(new_markdown_cell(
        "## 2. Univers optimisable\n\n"
        "Univers d'investissement, métriques par actif et statut qualité APT. Tous les actifs sont "
        "inclus dans l'optimisation et appartiennent à la poche optimisable."
    ))

    cells.append(new_code_cell(
        "pocket = result[\"03_Optimizable_Pocket\"]\n"
        "display(pocket)\n"
        "\n"
        "assert pocket[\"Included_in_Optimization\"].all()\n"
        "universe_assets = set(result[\"universe\"][\"asset_id\"])\n"
        "assert set(pocket[\"Asset\"]).issubset(universe_assets)"
    ))

    cells.append(new_markdown_cell(
        "## 3. Trois scénarios APT\n\n"
        "Les trois scénarios APT issus du notebook 01 permettent de tester si l'allocation "
        "additionnelle de 10 MD reste robuste en environnement prudent, central et optimiste. Les "
        "deltas par rapport au scénario central sont affichés pour repérer les actifs les plus "
        "sensibles aux hypothèses macro."
    ))

    cells.append(new_code_cell(
        "apt = result[\"04_APT_Scenarios\"]\n"
        "display(apt)\n"
        "\n"
        "display(Markdown(\n"
        "    \"_Les trois scénarios APT permettent de tester si l'allocation additionnelle de 10 MD reste \"\n"
        "    \"robuste en environnement prudent, central et optimiste. Le scénario central reste la \"\n"
        "    \"référence principale ; les scénarios prudent et optimiste servent de bornes._\"\n"
        "))"
    ))

    cells.append(new_markdown_cell(
        "## 4. Rendements actuels par scénario\n\n"
        "Pour chaque scénario APT, on calcule :\n\n"
        "- `R_OPT_CURRENT` = somme des poids actuels de la poche optimisable × rendements APT du scénario ;\n"
        "- `R_TOTAL_CURRENT_OR_PROXY` = proxy total avec la poche figée à rendement nul.\n\n"
        "Il n'y a pas de mélange entre rendement historique et rendement APT ; la référence est le rendement APT."
    ))

    cells.append(new_code_cell(
        "display(result[\"05b_Current_Returns\"])"
    ))

    cells.append(new_markdown_cell(
        "## 5. Rendement requis sur les 10 MD\n\n"
        "Rendement marginal requis pour atteindre la cible `TSR + 4 %` sur chaque scénario :\n\n"
        "- sur la poche optimisable : "
        "`R_REQUIRED = (TARGET × V_OPT_FINAL − R_OPT_CURRENT × V_OPT_CURRENT) / ADDITIONAL_BUDGET` ;\n"
        "- sur le portefeuille total : formule analogue.\n\n"
        "Le `Best_Achievable_R_Additional` est le rendement maximal atteignable sur les 10 MD sous contraintes."
    ))

    cells.append(new_code_cell(
        "display(result[\"05_Required_Returns\"])\n"
        "\n"
        "any_infeasible = result[\"05_Required_Returns\"][[\"Feasibility_Status_Opt\", \"Feasibility_Status_Total\"]].eq(\"INFEASIBLE\").any().any()\n"
        "if any_infeasible:\n"
        "    display(Markdown(\n"
        "        \"**TARGET_NOT_REACHED** sous au moins un scénario : le rendement marginal requis sur les 10 MD \"\n"
        "        \"est supérieur aux rendements réalistes disponibles dans l'univers d'investissement. \"\n"
        "        \"Le résultat traduit un effet de taille de l'enveloppe et non un échec du solveur.\"\n"
        "    ))"
    ))

    cells.append(new_markdown_cell(
        "## 6. Résultats par modèle et par scénario\n\n"
        "Pour chaque scénario APT, les modèles d'allocation suivants sont exécutés (Max Sharpe "
        "volontairement exclu) :\n\n"
        "1. **Prorata** de la poche optimisable actuelle (benchmark) ;\n"
        "2. **Minimum Variance** ;\n"
        "3. **Mean-Variance** avec trois niveaux d'aversion (λ = 2, 5, 10) ;\n"
        "4. **Maximum Return** sous contraintes (borne supérieure agressive, **pas** recommandation finale) ;\n"
        "5. **Mean-CVaR 95 %** ;\n"
        "6. **Risk Parity** ;\n"
        "7. **Monte Carlo** : Max Return, Min Volatility, Min CVaR, Best Scoring ;\n"
        "8. **Scoring multicritère final** (rendement, volatilité, CVaR, diversification, conformité, etc.).\n\n"
        "Maximum Return est présenté comme **borne supérieure** ; la recommandation finale provient du scoring."
    ))

    cells.append(new_code_cell(
        "cross = result[\"07_Model_Results_By_Scenario\"]\n"
        "display(cross)"
    ))

    cells.append(new_markdown_cell(
        "### Impact sur la poche optimisable et sur le portefeuille global"
    ))

    cells.append(new_code_cell(
        "display(result[\"08_Impact_Optimizable_Pocket\"])\n"
        "display(result[\"09_Impact_Total_Portfolio\"])"
    ))

    cells.append(new_markdown_cell(
        "## 7. Allocations détaillées des 10 MD par modèle\n\n"
        "Pour chaque modèle et chaque scénario, on affiche les actifs ayant reçu une allocation "
        "significative (≥ 0.01 %). Tous les actifs alloués appartiennent à la poche optimisable."
    ))

    cells.append(new_code_cell(
        "alloc = result[\"06_Allocations_10MD\"]\n"
        "alloc_display = alloc.loc[alloc[\"display_in_main_tables\"]].copy()\n"
        "display(alloc_display[[\n"
        "    \"Scenario\", \"Model\", \"asset_name\", \"asset_class\", \"weight_10md\",\n"
        "    \"amount_allocated_DT\", \"final_weight_opt\", \"final_weight_total\",\n"
        "]].head(80))\n"
        "\n"
        "assert (alloc[\"weight_10md\"] >= -1e-10).all()\n"
        "for (scenario, model), group in alloc.groupby([\"Scenario\", \"Model\"]):\n"
        "    if model.startswith(\"Monte_Carlo_\") and model not in result[\"by_scenario\"][scenario][\"results_models\"][\"Model\"].tolist():\n"
        "        continue\n"
        "    s = float(group[\"weight_10md\"].sum())\n"
        "    assert abs(s - 1.0) <= 1e-6, f\"{scenario}/{model}: weights sum = {s}\"\n"
        "    total_dt = float(group[\"amount_allocated_DT\"].sum())\n"
        "    assert abs(total_dt - ADDITIONAL_BUDGET) <= 1.0, f\"{scenario}/{model}: amount = {total_dt}\"\n"
        "    asset_classes = set(group.loc[group[\"weight_10md\"] > 0, \"asset_type\"])\n"
        "    assert asset_classes.issubset({\"government_bond\", \"corporate_bond\", \"listed_equity\"}), \\\n"
        "        f\"Actif non optimisable détecté pour {scenario}/{model}\""
    ))

    cells.append(new_markdown_cell(
        "## 8. Recommandation finale par scénario\n\n"
        "Le scoring multicritère choisit, pour chaque scénario APT, l'allocation institutionnellement "
        "défendable. Maximum Return est conservé comme borne supérieure mais pénalisé par le scoring."
    ))

    cells.append(new_code_cell(
        "display(result[\"14_Final_Recommendation\"])\n"
        "display(result[\"14b_Status\"])"
    ))

    cells.append(new_markdown_cell(
        "## 9. Contraintes réglementaires sur le portefeuille global final\n\n"
        "Les contraintes d'optimisation s'appliquent à l'allocation des 10 MD, mais les contraintes "
        "réglementaires sont vérifiées sur le **portefeuille global final** (portefeuille actuel + "
        "allocation des 10 MD). Statuts : `PASSED`, `FAILED`, `NON_TESTABLE_DATA_MISSING`."
    ))

    cells.append(new_code_cell(
        "central_reco = result[\"by_scenario\"][\"APT_Central\"][\"recommended_model\"]\n"
        "regulatory = result[\"10_Regulatory_Checks\"]\n"
        "reg_central = regulatory.loc[\n"
        "    regulatory[\"Scenario\"].eq(\"APT_Central\") & regulatory[\"model\"].eq(central_reco)\n"
        "]\n"
        "display(reg_central[[\n"
        "    \"Scenario\", \"model\", \"Constraint\", \"Status\", \"exposition_avant\",\n"
        "    \"exposition_apres\", \"seuil\", \"marge_apres\", \"testable\",\n"
        "]])\n"
        "\n"
        "testable_central = reg_central.loc[reg_central[\"testable\"]]\n"
        "assert not testable_central[\"Status\"].eq(\"FAILED\").any()\n"
        "\n"
        "display(Markdown(\n"
        "    \"Aucune violation n'est détectée sur les contraintes testables. Certaines contraintes \"\n"
        "    \"réglementaires restent non testables faute de données détaillées (référentiel émetteur, \"\n"
        "    \"capital social, détail OPCVM, actions non cotées).\"\n"
        "))"
    ))

    cells.append(new_markdown_cell(
        "## 10. Monte Carlo (sans Max Sharpe)\n\n"
        "Sélection Monte Carlo distincte pour chaque scénario : Max Return, Min Volatility, Min CVaR, "
        "Best Scoring."
    ))

    cells.append(new_code_cell(
        "mc_selected = result[\"11b_Monte_Carlo_Selected\"]\n"
        "display(mc_selected[[\n"
        "    \"Scenario\", \"Selection\", \"portfolio_id\", \"R_additional\", \"R_opt_final\",\n"
        "    \"R_total_final\", \"Volatility_additional\", \"CVaR_95\", \"Target_Status\",\n"
        "    \"Selection_Note\",\n"
        "]])\n"
        "\n"
        "for scenario in SCENARIO_ORDER:\n"
        "    assert (result[\"by_scenario\"][scenario][\"monte_carlo\"].shape[0] >= 15_000)"
    ))

    cells.append(new_markdown_cell(
        "### VaR / CVaR / Sharpe — précautions\n\n"
        "Le ratio de Sharpe est affiché à titre indicatif. Il **n'est pas utilisé comme critère principal "
        "de décision**, compte tenu du lissage possible de la volatilité sur certains actifs obligataires "
        "revalorisés par modèle.\n\n"
        "Les VaR/CVaR à 95 % sont calculées sur les **rendements périodiques** (jamais sur des rendements "
        "déjà annualisés). Une VaR ou une CVaR proche de zéro pour un portefeuille contenant des actions "
        "doit être interprétée avec prudence : elle peut refléter une période courte ou un lissage des séries."
    ))

    cells.append(new_markdown_cell(
        "## 11. Analyse de sensibilité\n\n"
        "Deux questions complémentaires :\n\n"
        "1. **Rendement final** si les 10 MD rapportent 8 %, 10 %, 12 %, 15 %, 20 %, 30 %, 40 %, 50 %, 60 % ;\n"
        "2. **Budget additionnel requis** pour atteindre la cible si le rendement additionnel réaliste est 10 %, 12 %, 15 %, 20 %.\n\n"
        "Si le rendement additionnel hypothétique est inférieur ou égal à la cible, la formule est non définie."
    ))

    cells.append(new_code_cell(
        "sensitivity = result[\"13_Sensitivity_Analysis\"]\n"
        "display(sensitivity)"
    ))

    cells.append(new_markdown_cell(
        "## 12. Warnings qualité"
    ))

    cells.append(new_code_cell(
        "display(result[\"Warnings_Quality\"])"
    ))

    cells.append(new_markdown_cell(
        "## 13. Graphiques\n\n"
        "Dix graphiques utiles produits sous Plotly, exportés en HTML dans `exports/figures/notebook_03/`."
    ))

    cells.append(new_code_cell(
        "exported_figures = []\n"
        "\n"
        "def show_and_export_fig(fig, name, output_dir=FIGURE_DIR):\n"
        "    fig.show(renderer=\"notebook_connected\")\n"
        "    path = output_dir / f\"{name}.html\"\n"
        "    fig.write_html(path, include_plotlyjs=\"cdn\", full_html=True)\n"
        "    exported_figures.append(path)\n"
        "    return path\n"
        "\n"
        "for old in FIGURE_DIR.glob(\"*.html\"):\n"
        "    old.unlink()"
    ))

    cells.append(new_markdown_cell(
        "### Graphique 1 — Composition actuelle de la poche optimisable\n\n"
        "Mise en évidence des poids actuels par classe d'actif et par titre."
    ))

    cells.append(new_code_cell(
        "fig = px.bar(\n"
        "    result[\"03_Optimizable_Pocket\"],\n"
        "    x=\"Asset\", y=\"Poids actuel poche optimisable\", color=\"Classe\",\n"
        "    title=\"Composition actuelle de la poche optimisable\",\n"
        "    labels={\"Asset\": \"Actif\", \"Poids actuel poche optimisable\": \"Poids\"},\n"
        ")\n"
        "fig.update_layout(xaxis_tickangle=-45)\n"
        "show_and_export_fig(fig, \"01_composition_poche_optimisable\")"
    ))

    cells.append(new_markdown_cell(
        "### Graphique 2 — Allocation des 10 MD par modèle recommandé et par scénario\n\n"
        "Pour chaque scénario APT, on affiche la répartition des 10 MD selon le modèle recommandé."
    ))

    cells.append(new_code_cell(
        "rows = []\n"
        "for scenario in SCENARIO_ORDER:\n"
        "    scn = result[\"by_scenario\"][scenario]\n"
        "    model = scn[\"recommended_model\"]\n"
        "    sub = result[\"06_Allocations_10MD\"]\n"
        "    sub = sub.loc[\n"
        "        sub[\"Scenario\"].eq(scenario)\n"
        "        & sub[\"Model\"].eq(model)\n"
        "        & sub[\"display_in_main_tables\"]\n"
        "    ]\n"
        "    rows.append(sub)\n"
        "alloc_reco = pd.concat(rows, ignore_index=True)\n"
        "fig = px.bar(\n"
        "    alloc_reco, x=\"asset_name\", y=\"amount_allocated_DT\", color=\"Scenario\", barmode=\"group\",\n"
        "    title=\"Allocation des 10 MD — modèle recommandé par scénario\",\n"
        "    labels={\"asset_name\": \"Actif\", \"amount_allocated_DT\": \"Montant alloué (TND)\"},\n"
        ")\n"
        "fig.update_layout(xaxis_tickangle=-45)\n"
        "show_and_export_fig(fig, \"02_allocation_10md_par_scenario\")"
    ))

    cells.append(new_markdown_cell(
        "### Graphique 3 — Rendement additionnel des 10 MD par modèle et scénario"
    ))

    cells.append(new_code_cell(
        "fig = px.bar(\n"
        "    cross, x=\"Model\", y=\"R_additional\", color=\"Scenario\", barmode=\"group\",\n"
        "    title=\"Rendement additionnel des 10 MD par modèle et scénario\",\n"
        "    labels={\"Model\": \"Modèle\", \"R_additional\": \"Rendement additionnel\"},\n"
        ")\n"
        "fig.update_layout(xaxis_tickangle=-35)\n"
        "show_and_export_fig(fig, \"03_rendement_additionnel_par_modele_scenario\")"
    ))

    cells.append(new_markdown_cell(
        "### Graphique 4 — Rendement final de la poche optimisable vs cible"
    ))

    cells.append(new_code_cell(
        "fig = px.bar(\n"
        "    cross, x=\"Model\", y=\"R_opt_final\", color=\"Scenario\", barmode=\"group\",\n"
        "    title=\"Rendement final de la poche optimisable vs cible\",\n"
        ")\n"
        "fig.add_hline(y=TARGET_RETURN, line_dash=\"dash\", annotation_text=\"Cible\")\n"
        "fig.update_layout(xaxis_tickangle=-35)\n"
        "show_and_export_fig(fig, \"04_rendement_final_poche_vs_cible\")"
    ))

    cells.append(new_markdown_cell(
        "### Graphique 5 — Rendement final du portefeuille total vs cible"
    ))

    cells.append(new_code_cell(
        "fig = px.bar(\n"
        "    cross, x=\"Model\", y=\"R_total_final\", color=\"Scenario\", barmode=\"group\",\n"
        "    title=\"Rendement final du portefeuille total vs cible\",\n"
        ")\n"
        "fig.add_hline(y=TARGET_RETURN, line_dash=\"dash\", annotation_text=\"Cible\")\n"
        "fig.update_layout(xaxis_tickangle=-35)\n"
        "show_and_export_fig(fig, \"05_rendement_final_total_vs_cible\")"
    ))

    cells.append(new_markdown_cell(
        "### Graphique 6 — Écart à la cible par scénario et modèle"
    ))

    cells.append(new_code_cell(
        "gap = cross.melt(\n"
        "    id_vars=[\"Scenario\", \"Model\"], value_vars=[\"Gap_Opt\", \"Gap_Total\"],\n"
        "    var_name=\"Périmètre\", value_name=\"Gap\",\n"
        ")\n"
        "fig = px.bar(\n"
        "    gap, x=\"Model\", y=\"Gap\", color=\"Scenario\", barmode=\"group\", facet_row=\"Périmètre\",\n"
        "    title=\"Écart à la cible par scénario et modèle (Gap = R_final − Cible)\",\n"
        ")\n"
        "fig.update_layout(xaxis_tickangle=-35, height=620)\n"
        "show_and_export_fig(fig, \"06_gap_a_la_cible\")"
    ))

    cells.append(new_markdown_cell(
        "### Graphique 7 — Nuage Monte Carlo (sans Max Sharpe) par scénario"
    ))

    cells.append(new_code_cell(
        "mc = result[\"11_Monte_Carlo\"].copy()\n"
        "selected_ids = result[\"11b_Monte_Carlo_Selected\"][[\"Scenario\", \"portfolio_id\"]].dropna()\n"
        "selected_ids[\"portfolio_id\"] = selected_ids[\"portfolio_id\"].astype(int)\n"
        "selected_set = set(zip(selected_ids[\"Scenario\"], selected_ids[\"portfolio_id\"]))\n"
        "mc[\"Selected\"] = [\n"
        "    (s, int(p)) in selected_set for s, p in zip(mc[\"Scenario\"], mc[\"portfolio_id\"])\n"
        "]\n"
        "fig = px.scatter(\n"
        "    mc, x=\"Volatility_additional\", y=\"R_additional\",\n"
        "    color=\"Scenario\", symbol=\"Selected\", opacity=0.35,\n"
        "    title=\"Monte Carlo — rendement vs volatilité (10 MD), portefeuilles sélectionnés mis en avant\",\n"
        ")\n"
        "fig.add_hline(y=TARGET_RETURN, line_dash=\"dash\", annotation_text=\"Cible TSR+4%\")\n"
        "show_and_export_fig(fig, \"07_monte_carlo_par_scenario\")"
    ))

    cells.append(new_markdown_cell(
        "### Graphique 8 — Risque vs rendement des modèles déterministes"
    ))

    cells.append(new_code_cell(
        "deterministic_models = [\n"
        "    \"Prorata_Current_Optimizable_Pocket\", \"Minimum_Variance\",\n"
        "    \"Mean_Variance_Aversion_2\", \"Mean_Variance_Aversion_5\", \"Mean_Variance_Aversion_10\",\n"
        "    \"Max_Return_Constraints\", \"Mean_CVaR\", \"Risk_Parity\",\n"
        "]\n"
        "det = cross.loc[cross[\"Model\"].isin(deterministic_models)]\n"
        "fig = px.scatter(\n"
        "    det, x=\"Volatility_additional\", y=\"R_additional\", color=\"Scenario\", symbol=\"Model\",\n"
        "    text=\"Model\",\n"
        "    title=\"Risque vs rendement des modèles déterministes (10 MD)\",\n"
        ")\n"
        "fig.update_traces(textposition=\"top center\")\n"
        "show_and_export_fig(fig, \"08_risque_vs_rendement_modeles\")"
    ))

    cells.append(new_markdown_cell(
        "### Graphique 9 — Sensibilité du rendement final selon le rendement des 10 MD"
    ))

    cells.append(new_code_cell(
        "sens_ret = sensitivity.loc[sensitivity[\"Type\"].eq(\"Rendement final pour 10 MD\")]\n"
        "fig = px.line(\n"
        "    sens_ret, x=\"Hypothèse rendement additionnel\", y=\"Rendement final\",\n"
        "    color=\"Scenario\", line_dash=\"Périmètre\", markers=True,\n"
        "    title=\"Sensibilité du rendement final selon le rendement des 10 MD\",\n"
        ")\n"
        "fig.add_hline(y=TARGET_RETURN, line_dash=\"dash\", annotation_text=\"Cible TSR+4%\")\n"
        "show_and_export_fig(fig, \"09_sensibilite_rendement_final\")"
    ))

    cells.append(new_markdown_cell(
        "### Graphique 10 — Montant additionnel requis selon l'hypothèse de rendement"
    ))

    cells.append(new_code_cell(
        "sens_amt = sensitivity.loc[\n"
        "    sensitivity[\"Type\"].eq(\"Montant additionnel requis\")\n"
        "].dropna(subset=[\"Montant requis\"])\n"
        "fig = px.bar(\n"
        "    sens_amt, x=\"Hypothèse rendement additionnel\", y=\"Montant requis\",\n"
        "    color=\"Scenario\", barmode=\"group\", facet_row=\"Périmètre\",\n"
        "    title=\"Budget additionnel requis pour atteindre la cible TSR+4%\",\n"
        ")\n"
        "fig.update_layout(height=620)\n"
        "show_and_export_fig(fig, \"10_budget_additionnel_requis\")\n"
        "\n"
        "print(f\"{len(exported_figures)} figures exportées dans {FIGURE_DIR}\")"
    ))

    cells.append(new_markdown_cell(
        "## 14. Export Excel et contrôle final\n\n"
        "Le classeur Excel est exporté avec les 15 feuilles attendues : hypothèses, portefeuille, "
        "poche optimisable, scénarios APT, rendements requis, allocations, résultats par scénario, "
        "impacts, contraintes, Monte Carlo, scoring, sensibilité, recommandation, conclusion."
    ))

    cells.append(new_code_cell(
        "from maghrebia_quant.allocation_10md import _build_final_control_multi\n"
        "\n"
        "figures_ok = len(exported_figures) >= 10 and all(p.exists() for p in exported_figures)\n"
        "excel_path = export_multi_scenario_analysis(result, EXCEL_PATH)\n"
        "excel_ok = excel_path.exists()\n"
        "\n"
        "result[\"Controle_Final\"] = _build_final_control_multi(\n"
        "    result, result[\"by_scenario\"], CONFIG,\n"
        "    figures_exported=figures_ok, excel_exported=excel_ok,\n"
        ")\n"
        "display(result[\"Controle_Final\"])\n"
        "display(Markdown(f\"Classeur Excel : `{EXCEL_PATH.relative_to(PROJECT_ROOT)}`\"))\n"
        "display(Markdown(f\"Figures HTML : `{FIGURE_DIR.relative_to(PROJECT_ROOT)}`\"))"
    ))

    cells.append(new_markdown_cell(
        "## 15. Conclusion finale\n\n"
        "Conclusion structurée en 8 points + formulation institutionnelle, conforme à la consigne du PFE."
    ))

    cells.append(new_code_cell(
        "display(result[\"15_Conclusion\"])"
    ))

    cells.append(new_code_cell(
        "central = result[\"by_scenario\"][\"APT_Central\"]\n"
        "reco_metrics = central[\"results_models\"].loc[\n"
        "    central[\"results_models\"][\"Model\"].eq(central[\"recommended_model\"])\n"
        "].iloc[0]\n"
        "\n"
        "control = result[\"Controle_Final\"]\n"
        "technical_status = control.loc[control[\"Contrôle\"].eq(\"Technical_Status\"), \"Status\"].iloc[0]\n"
        "global_status = control.loc[control[\"Contrôle\"].eq(\"Statut global\"), \"Status\"].iloc[0]\n"
        "\n"
        "status_lines = []\n"
        "for scenario in SCENARIO_ORDER:\n"
        "    scn = result[\"by_scenario\"][scenario]\n"
        "    reco = scn[\"results_models\"].loc[scn[\"results_models\"][\"Model\"].eq(scn[\"recommended_model\"])].iloc[0]\n"
        "    reached = reco[\"Target_Opt_Reached\"] == \"YES\" and reco[\"Target_Total_Reached\"] == \"YES\"\n"
        "    status_lines.append(\n"
        "        f\"- **{scenario}** — modèle recommandé `{scn['recommended_model']}` — \"\n"
        "        f\"`Target_Status = {'TARGET_REACHED' if reached else 'TARGET_NOT_REACHED'}` — \"\n"
        "        f\"R_additional = {reco['R_additional']:.2%} ; R_opt_final = {reco['R_opt_final']:.2%} ; \"\n"
        "        f\"R_total_final = {reco['R_total_final']:.2%} ; Gap_Opt = {reco['Gap_Opt']:+.2%} ; \"\n"
        "        f\"Gap_Total = {reco['Gap_Total']:+.2%}.\"\n"
        "    )\n"
        "\n"
        "display(Markdown(\n"
        "    f\"**Technical_Status** = `{technical_status}`  \\n\"\n"
        "    f\"**Statut global** = `{global_status}`  \\n\\n\"\n"
        "    + \"\\n\".join(status_lines)\n"
        "    + (\n"
        "        \"\\n\\n_Le notebook est techniquement valide. L'objectif TSR + 4 % n'est pas atteint sous \"\n"
        "        \"toutes les hypothèses retenues : la non-atteinte traduit un effet de taille et un \"\n"
        "        \"rendement marginal requis supérieur aux rendements réalistes disponibles, pas une \"\n"
        "        \"défaillance technique de l'analyse._\"\n"
        "    )\n"
        "))"
    ))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.13.5",
    }

    out_path = Path("notebooks/03_allocation_supplementaire_10MD.ipynb")
    nbformat.write(nb, out_path.as_posix())
    print(f"Notebook écrit : {out_path}")
    return out_path


if __name__ == "__main__":
    main()
