from pathlib import Path

import numpy as np
import pandas as pd

from src.data import TARGET, clean_features, load_dataset

DATA_DIR = Path(__file__).parents[1] / "data"


def test_real_csv_structure_is_reported_and_safe_record_is_recovered():
    train, test, reports = load_dataset(DATA_DIR)

    assert train.shape[0] == 950
    assert test.shape[0] == 50
    assert reports[0].malformed_lines == (952,)
    assert reports[0].repaired_lines == (69,)
    assert reports[1].malformed_lines == ()
    assert reports[1].repaired_lines == ()

    recovered_rows = train.loc[train["ID"] == 68]
    assert len(recovered_rows) == 1
    recovered = recovered_rows.iloc[0]
    assert isinstance(recovered, pd.Series)
    assert np.isnan(recovered["eta_anni"])
    assert np.isnan(recovered["peso_kg"])
    assert recovered["sesso"] == "femmina"
    assert recovered[TARGET] == "Bengala"


def test_cleaning_normalises_domains_and_removes_implausible_numbers():
    dirty = pd.DataFrame(
        {
            "ID": [1],
            "eta_anni": ["52"],
            "peso_kg": ["646"],
            "sesso": [" F "],
            "lunghezza_pelo": ["arcobaleno"],
            "colore_mantello": ["Blue Point"],
            "livello_attivita": ["molto attivo"],
            "frequenza_miagolio": ["alto"],
            "sterilizzato": ["forse"],
            "patologia": [""],
            "classe": ["molto_attivo"],
        }
    )
    cleaned = clean_features(dirty).iloc[0]

    assert cleaned["sesso"] == "femmina"
    assert cleaned["lunghezza_pelo"] == "sconosciuto"
    assert cleaned["colore_mantello"] == "blue_point"
    assert cleaned["livello_attivita"] == "molto_attivo"
    assert cleaned["sterilizzato"] == "sconosciuto"
    assert cleaned["patologia"] == "sconosciuto"
    assert np.isnan(cleaned["eta_anni"])
    assert np.isnan(cleaned["peso_kg"])
    assert "eta_anni_anomalo" not in cleaned.index
    assert "peso_kg_anomalo" not in cleaned.index
