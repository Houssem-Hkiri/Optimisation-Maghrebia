"""Configuration stricte des operations sur capital, dividendes et overrides obligataires."""

CORPORATE_ACTIONS = [
    {
        "asset_id": "ATTIJARI_BANK",
        "bvmt_code": "TN0001600154",
        "asset_name_exact": "ATTIJARI BANK",
        "action_type": "price_adjustment",
        "effective_date": "2025-04-30",
        "price_adjustment_factor": 21 / 25,
        "quantity_adjustment_factor": 25 / 21,
        "source": "BVMT - augmentation de capital du 30/04/2025",
        "comment": "Ajustement Attijari Bank uniquement, sans application a Attijari Leasing ni aux droits/rompus.",
    },
    {
        "asset_id": "DELICE_HOLDING",
        "bvmt_code": "TN0007670011",
        "asset_name_exact": "DELICE HOLDING",
        "action_type": "price_adjustment",
        "effective_date": "2025-08-01",
        "price_adjustment_factor": 0.5,
        "quantity_adjustment_factor": 2.0,
        "source": "BVMT - communique de presse du 28/07/2025",
        "comment": "Split Delice Holding uniquement.",
    },
    {
        "asset_id": "AMEN_BANK",
        "bvmt_code": "TN0003400058",
        "asset_name_exact": "AMEN BANK",
        "action_type": "dividend",
        "effective_date": "2025-05-20",
        "amount": 3.300,
        "price_adjustment_factor": 1.0,
        "quantity_adjustment_factor": 1.0,
        "source": "Tunisie Clearing - calendrier dividendes 2025",
        "comment": "Dividende suivi pour lecture du rendement prix ; aucun ajustement de prix n'est applique.",
    },
    {
        "asset_id": "BT",
        "bvmt_code": "TN0002200053",
        "asset_name_exact": "BT",
        "action_type": "dividend",
        "effective_date": "2025-05-07",
        "amount": 0.350,
        "price_adjustment_factor": 1.0,
        "quantity_adjustment_factor": 1.0,
        "source": "Tunisie Clearing - calendrier dividendes 2025",
        "comment": "Dividende suivi pour lecture du rendement prix ; aucun ajustement de prix n'est applique.",
    },
    {
        "asset_id": "BIAT",
        "bvmt_code": "TN0001800457",
        "asset_name_exact": "BIAT",
        "action_type": "dividend",
        "effective_date": "2025-05-12",
        "amount": 6.000,
        "price_adjustment_factor": 1.0,
        "quantity_adjustment_factor": 1.0,
        "source": "BVMT / Managers - dividende BIAT 2025",
        "comment": "Dividende BIAT 2025 de 6,000 DT ; rendement prix seul, aucun ajustement de prix.",
    },
]


CORPORATE_BOND_SPREAD_OVERRIDES = {
    "TN0003100872": {
        "spread_decimal": None,
        "spread_date": None,
        "comment": "Spread BNA SUB individuel - prime subordonnee non captee par la mediane sectorielle BANKING.",
    },
    "TNDE9EH7SA12": {
        "spread_decimal": None,
        "spread_date": None,
        "comment": "Spread AB SUB individuel.",
    },
}


CORPORATE_BOND_METADATA_OVERRIDES = {
    "EO_ATL_2025_2": {
        "isin": "TN7XUIXDVQY8",
        "dirty_price_scale": "BASE_1",
        "comment": "ATL 2025-2 introduit en Bourse le 17/04/2026 ; ISIN utilise pour tracer la serie proxy 2025.",
    }
}
