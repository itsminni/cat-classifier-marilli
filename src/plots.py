"""Grafici per la consegna della challenge."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid", context="notebook")

CATEGORICAL_PALETTE = "colorblind"
SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "vlag_r"
PRIMARY_COLOR = sns.color_palette(CATEGORICAL_PALETTE)[0]


def _display_label(value: object) -> str:
    return str(value).replace("_", " ")


def _category_colors(values: pd.Series) -> dict[object, tuple[float, float, float]]:
    """Associa colori stabili alle categorie, indipendentemente dal loro ordine."""
    categories = sorted(values.dropna().unique())
    colors = sns.color_palette(CATEGORICAL_PALETTE, n_colors=len(categories))
    return dict(zip(categories, colors, strict=True))


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_target_distribution(train: pd.DataFrame, output: Path) -> None:
    counts = train["razza"].value_counts().sort_values()
    _, ax = plt.subplots(figsize=(9, 5))
    palette = _category_colors(train["razza"])
    colors = [palette[value] for value in counts.index]
    ax.barh([_display_label(value) for value in counts.index], counts.to_numpy(), color=colors)
    ax.set(
        title="Distribuzione delle razze nel training set",
        xlabel="Numero di record",
        ylabel="Razza",
    )
    for y_position, value in enumerate(counts.to_numpy()):
        ax.text(
            value + 2,
            y_position,
            f"{value} ({value / len(train):.1%})",
            va="center",
        )
    ax.set_xlim(0, counts.max() * 1.22)
    _save(output)


def plot_numeric_correlation(train: pd.DataFrame, output: Path) -> None:
    numeric_features = cast(pd.DataFrame, train[["eta_anni", "peso_kg"]]).rename(
        columns={"eta_anni": "Età (anni)", "peso_kg": "Peso (kg)"}
    )
    corr = numeric_features.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=DIVERGING_CMAP,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
    )
    plt.title("Correlazione tra le feature numeriche (dopo pulizia)")
    _save(output)


def plot_coat_target_relation(train: pd.DataFrame, output: Path) -> None:
    table = pd.crosstab(train["colore_mantello"], train["razza"], normalize="index")
    supports = train["colore_mantello"].value_counts()
    order = supports.index
    table = table.loc[order]
    plt.figure(figsize=(11, 9))
    ax = sns.heatmap(
        table,
        cmap=SEQUENTIAL_CMAP,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Quota per colore"},
    )
    ax.set_yticklabels(
        [f"{_display_label(value)} (n={supports[value]})" for value in order], rotation=0
    )
    ax.set_xticklabels([_display_label(value) for value in table.columns], rotation=0)
    plt.title("Relazione tra colore del mantello e razza")
    plt.xlabel("Razza")
    plt.ylabel("Colore del mantello")
    _save(output)


def plot_confusion(y_true: pd.Series, y_pred, output: Path) -> None:
    labels = sorted(y_true.unique())
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    # Le predizioni OOF permettono di annotare ogni cella sia con il numero di
    # esempi sia con la percentuale rispetto alla classe reale senza usare il test set.
    annotations = [
        [f"{count}\n{share:.0%}" for count, share in zip(row_count, row_share, strict=True)]
        for row_count, row_share in zip(counts, normalized, strict=True)
    ]
    supports = counts.sum(axis=1)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        normalized,
        annot=annotations,
        fmt="",
        cmap=SEQUENTIAL_CMAP,
        vmin=0,
        vmax=1,
        xticklabels=[_display_label(label) for label in labels],
        yticklabels=[
            f"{_display_label(label)} (n={support})"
            for label, support in zip(labels, supports, strict=True)
        ],
    )
    plt.title("Matrice di confusione OOF — conteggi e percentuali per classe")
    plt.xlabel("Razza prevista")
    plt.ylabel("Razza reale")
    plt.xticks(rotation=35, ha="right")
    _save(output)


def plot_model_comparison(results: pd.DataFrame, output: Path) -> None:
    ordered = results.sort_values("accuracy_media")
    competitive = ordered[~ordered["modello"].str.startswith("Dummy")]
    # La vista completa rende evidente il vantaggio sul baseline.
    height = max(5, 0.42 * len(ordered))
    figure, axes = plt.subplots(1, 2, figsize=(15, height), width_ratios=[1, 1.35])
    for axis, data in zip(axes, [ordered, competitive], strict=True):
        axis.errorbar(
            data["accuracy_media"],
            data["modello"],
            xerr=data["accuracy_std"],
            fmt="o",
            capsize=5,
            color=PRIMARY_COLOR,
        )
        for _, row in data.iterrows():
            axis.annotate(
                f"{row['accuracy_media']:.2%}",
                (row["accuracy_media"], row["modello"]),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=9,
            )
        axis.set_xlabel("Accuracy media ± dev. standard")
        axis.set_ylabel("Modello")
    axes[0].set_xlim(0, 1.04)
    lower = max(0, competitive["accuracy_media"].min() - 0.02)
    upper = min(1.005, competitive["accuracy_media"].max() + 0.015)
    axes[1].set_xlim(lower, upper)
    axes[0].set_title("Vista completa")
    axes[1].set_title("Zoom sui modelli competitivi")
    figure.suptitle("Confronto con cross-validation stratificata a 5 fold")
    _save(output)


def plot_class_metrics(report: pd.DataFrame, output: Path) -> None:
    """Confronta precision, recall e F1 per ogni classe reale."""
    classes = report.loc[~report.index.str.contains("avg|accuracy")].copy()
    classes.index = classes.index.map(_display_label)
    long = (
        classes[["precision", "recall", "f1-score"]]
        .rename(columns={"f1-score": "F1"})
        .rename(columns=str.capitalize)
        .reset_index(names="classe")
        .melt(id_vars="classe", var_name="metrica", value_name="valore")
    )
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=long,
        x="classe",
        y="valore",
        hue="metrica",
        palette=CATEGORICAL_PALETTE,
    )
    ax.set(
        title="Metriche OOF per classe",
        xlabel="Razza",
        ylabel="Punteggio",
        ylim=(0, 1.05),
    )
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Metrica", loc="center left", bbox_to_anchor=(1.01, 0.5))
    _save(output)


def plot_feature_importance(importance: pd.DataFrame, output: Path) -> None:
    """Mostra l'importanza globale calcolata dal modello finale."""
    ordered = importance.sort_values("importanza").copy()
    ordered["feature"] = ordered["feature"].str.replace("_", " ")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=ordered, x="importanza", y="feature", color=PRIMARY_COLOR)
    ax.set(
        title="Importanza globale delle feature — CatBoost",
        xlabel="Importanza (%)",
        ylabel="Feature",
    )
    _save(output)


def plot_numeric_relationship(train: pd.DataFrame, output: Path) -> None:
    """Mostra la relazione tra le due feature numeriche."""
    display_data = train.copy()
    display_data["razza"] = display_data["razza"].map(_display_label)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=display_data,
        x="eta_anni",
        y="peso_kg",
        hue="razza",
        palette=_category_colors(display_data["razza"]),
        alpha=0.68,
        s=38,
    )
    plt.title("Relazione tra età e peso, colorata per razza")
    plt.xlabel("Età (anni)")
    plt.ylabel("Peso (kg)")
    plt.legend(title="Razza", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(output)
