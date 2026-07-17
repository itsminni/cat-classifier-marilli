"""Utilità per il caricamento e la pulizia dei dati indipendente dal target."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "razza"
ID_COLUMN = "ID"
NUMERIC_COLUMNS = ["eta_anni", "peso_kg"]
CATEGORICAL_COLUMNS = [
    "sesso",
    "lunghezza_pelo",
    "colore_mantello",
    "livello_attivita",
    "frequenza_miagolio",
    "sterilizzato",
    "patologia",
    "classe",
]
FEATURE_COLUMNS = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]
TRAIN_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET]
TEST_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS]


@dataclass(frozen=True)
class LoadReport:
    """Esito della validazione strutturale di un file CSV."""

    source: Path
    physical_rows: int
    valid_rows: int
    malformed_lines: tuple[int, ...]
    repaired_lines: tuple[int, ...]


def _repair_ambiguous_numeric_record(
    row: list[str], expected_columns: list[str]
) -> list[str] | None:
    """Recupera un record con i due campi numerici uniti o uno dei due assente.

    Il training set fornito contiene un record di questo tipo: dopo l'ID compare
    un solo valore numerico ambiguo, seguito da tutti i campi
    categorici. I valori non ambigui vengono conservati, mentre
    *entrambi* i campi numerici sono segnati come mancanti.
    """
    known_sex_tokens = {"maschio", "femmina", "m", "f", "femminuccia", "trans"}
    is_train_record = expected_columns == TRAIN_COLUMNS
    if (
        is_train_record
        and len(row) == len(expected_columns) - 1
        and len(row) >= 3
        and row[2].strip().lower() in known_sex_tokens
    ):
        try:
            int(row[0])
            float(row[1])
        except ValueError:
            return None
        # Si preservano solo i campi interpretabili con certezza.
        return [row[0], "", "", *row[2:]]
    return None


def load_csv_strict(
    path: str | Path, expected_columns: list[str]
) -> tuple[pd.DataFrame, LoadReport]:
    """Carica solo record validi e segnala le righe fisiche malformate.

    Un record malformato non viene ricostruito per tentativi né traslato.
    """
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows or rows[0] != expected_columns:
        actual = rows[0] if rows else []
        raise ValueError(f"Unexpected schema in {path}: {actual}")

    valid, malformed, repaired = [], [], []
    for line_number, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) == len(expected_columns):
            valid.append(row)
            continue

        recovered = _repair_ambiguous_numeric_record(row, expected_columns)
        if recovered is None:
            malformed.append(line_number)
        else:
            valid.append(recovered)
            repaired.append(line_number)

    frame = pd.DataFrame(valid, columns=expected_columns)
    report = LoadReport(path, len(rows) - 1, len(frame), tuple(malformed), tuple(repaired))
    return frame, report


def _normalise_token(value: object) -> str:
    """Restituisce una categoria stabile uniformando i valori scalari mancanti."""
    is_float_nan = isinstance(value, (float, np.floating)) and bool(np.isnan(value))
    if value is None or value is pd.NA or value is pd.NaT or is_float_nan:
        return "sconosciuto"
    if not str(value).strip():
        return "sconosciuto"
    return "_".join(str(value).strip().lower().split())


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Applica una pulizia deterministica delle feature, indipendente dal target."""
    cleaned = frame.copy()
    # La pulizia non consulta mai ``razza`` in quanto target.
    numeric_ids = pd.Series(
        pd.to_numeric(cleaned[ID_COLUMN], errors="raise"),
        index=cleaned.index,
    )
    cleaned[ID_COLUMN] = numeric_ids.astype(int)

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    # I valori fuori da limiti plausibili vengono trattati come mancanti.
    bounds = {"eta_anni": (0.0, 30.0), "peso_kg": (0.5, 20.0)}
    for column, (lower, upper) in bounds.items():
        out_of_bounds = cleaned[column].notna() & ~cleaned[column].between(
            lower, upper, inclusive="both"
        )
        cleaned.loc[out_of_bounds, column] = np.nan

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].map(_normalise_token)

    cleaned["sesso"] = cleaned["sesso"].replace(
        {"m": "maschio", "f": "femmina", "femminuccia": "femmina"}
    )
    # Le categorie impossibili o corrotte confluiscono in ``sconosciuto``.
    allowed_domains = {
        "sesso": {"maschio", "femmina"},
        "lunghezza_pelo": {"corto", "semilungo", "lungo"},
        "livello_attivita": {"sedentario", "attivo", "molto_attivo"},
        "frequenza_miagolio": {"basso", "medio", "alto"},
        "sterilizzato": {"si", "no"},
        "classe": {"sedentario", "attivo", "molto_attivo"},
    }
    for column, allowed in allowed_domains.items():
        cleaned[column] = cleaned[column].where(cleaned[column].isin(tuple(allowed)), "sconosciuto")
    cleaned["colore_mantello"] = cleaned["colore_mantello"].replace({"arcobaleno": "sconosciuto"})
    return cleaned


def load_dataset(
    data_dir: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[LoadReport, LoadReport]]:
    """Carica e pulisce train e test preservando l'ordine del test set."""
    data_dir = Path(data_dir)
    train, train_report = load_csv_strict(data_dir / "train.csv", TRAIN_COLUMNS)
    test, test_report = load_csv_strict(data_dir / "test.csv", TEST_COLUMNS)
    expected_incomplete_train_lines = {952}
    unexpected_train_lines = set(train_report.malformed_lines) - expected_incomplete_train_lines
    if unexpected_train_lines or test_report.malformed_lines:
        raise ValueError("Dataset contains unexpected malformed CSV records")
    train = clean_features(train)
    test = clean_features(test)
    # Grafia e maiuscole del target restano identiche a quelle fornite, perché
    # devono essere riportate senza alterazioni in ``predictions.csv``.
    train[TARGET] = train[TARGET].str.strip()
    if bool(train[TARGET].eq("").to_numpy(dtype=bool).any()):
        raise ValueError("Training targets must not be blank")
    train_has_duplicate_ids = bool(train[ID_COLUMN].duplicated().to_numpy(dtype=bool).any())
    test_has_duplicate_ids = bool(test[ID_COLUMN].duplicated().to_numpy(dtype=bool).any())
    if train_has_duplicate_ids or test_has_duplicate_ids:
        raise ValueError("ID values must be unique within each file")
    train_has_nonpositive_ids = bool(train[ID_COLUMN].le(0).to_numpy(dtype=bool).any())
    test_has_nonpositive_ids = bool(test[ID_COLUMN].le(0).to_numpy(dtype=bool).any())
    if train_has_nonpositive_ids or test_has_nonpositive_ids:
        raise ValueError("ID values must be positive")
    return train, test, (train_report, test_report)
