import os
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, send_file
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

SEED = 42
DATA_PATH = "Parkinsons-Telemonitoring-ucirvine.csv"
MODEL_PATH = os.path.join("model", "pipeline_parkinsons.pkl")
CHART_DIR = os.path.join("static", "charts")
TARGET = "total_updrs"

FEATURES = ["age", "hnr", "dfa", "sex", "ppe", "rpde"]

FIELD_INFO = {
    "age": {
        "label": "Edad",
        "unit": "anos",
        "placeholder": "Ejemplo: 65",
        "min": 18,
        "max": 100,
        "step": "1",
    },
    "hnr": {
        "label": "HNR",
        "unit": "relacion armonicos/ruido",
        "placeholder": "Ejemplo: 21.5",
        "min": 0,
        "max": 45,
        "step": "0.001",
    },
    "dfa": {
        "label": "DFA",
        "unit": "analisis de fluctuacion",
        "placeholder": "Ejemplo: 0.72",
        "min": 0,
        "max": 1.5,
        "step": "0.001",
    },
    "ppe": {
        "label": "PPE",
        "unit": "entropia del periodo de tono",
        "placeholder": "Ejemplo: 0.19",
        "min": 0,
        "max": 1,
        "step": "0.001",
    },
    "rpde": {
        "label": "RPDE",
        "unit": "entropia de periodo de recurrencia",
        "placeholder": "Ejemplo: 0.55",
        "min": 0,
        "max": 1,
        "step": "0.001",
    },
}

NOTEBOOK_METRICS = {
    "cv_r2": 0.9064,
    "cv_mae": 2.3337,
    "cv_r2_std": 0.0065,
    "cv_r2_var": 0.000043,
}


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df["sex"] = df["sex"].astype(int)
    return df


def make_pipeline():
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, FEATURES)],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_charts(y_test, y_pred, metrics):
    os.makedirs(CHART_DIR, exist_ok=True)
    scatter_path = os.path.join(CHART_DIR, "real_vs_predicho.png")
    residuals_path = os.path.join(CHART_DIR, "residuos.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred, alpha=0.42, s=22, color="#227c9d", label="Predicciones")
    lim_min = min(y_test.min(), y_pred.min()) - 1
    lim_max = max(y_test.max(), y_pred.max()) + 1
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color="#d1495b",
        linestyle="--",
        linewidth=1.5,
        label="Prediccion perfecta",
    )
    ax.set_xlabel("Valores reales (total_updrs)")
    ax.set_ylabel("Valores predichos (total_updrs)")
    ax.set_title(
        f"Real vs predicho - Random Forest S1\n"
        f"R2={metrics['split_r2']:.4f}  MAE={metrics['split_mae']:.4f}"
    )
    ax.legend()
    ax.grid(alpha=0.28)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    residuals = y_test.to_numpy() - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.42, s=22, color="#227c9d")
    axes[0].axhline(0, color="#d1495b", linestyle="--", linewidth=1.4)
    axes[0].axhline(
        residuals.mean(),
        color="#edae49",
        linestyle="-.",
        linewidth=1,
        label=f"Media: {residuals.mean():.4f}",
    )
    axes[0].set_xlabel("Valores predichos")
    axes[0].set_ylabel("Residuos (real - predicho)")
    axes[0].set_title("Residuos vs predichos")
    axes[0].legend()
    axes[0].grid(alpha=0.28)

    axes[1].hist(residuals, bins=30, color="#227c9d", edgecolor="white", alpha=0.88)
    axes[1].axvline(0, color="#d1495b", linestyle="--", linewidth=1.4, label="Residuo = 0")
    axes[1].axvline(
        residuals.mean(),
        color="#edae49",
        linestyle="-.",
        linewidth=1,
        label=f"Media: {residuals.mean():.4f}",
    )
    axes[1].set_xlabel("Residuos (real - predicho)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribucion de residuos")
    axes[1].legend()

    fig.suptitle("Analisis de residuos - Random Forest S1")
    fig.tight_layout()
    fig.savefig(residuals_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def train_and_save_pipeline():
    df = load_dataset()
    x = df[FEATURES]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )

    eval_pipeline = make_pipeline()
    eval_pipeline.fit(x_train, y_train)
    y_pred = eval_pipeline.predict(x_test)

    metrics = {
        **NOTEBOOK_METRICS,
        "split_r2": float(r2_score(y_test, y_pred)),
        "split_mae": float(mean_absolute_error(y_test, y_pred)),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }

    save_charts(y_test, y_pred, metrics)

    final_pipeline = make_pipeline()
    final_pipeline.fit(x, y)

    bundle = {
        "pipeline": final_pipeline,
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(bundle, MODEL_PATH)
    return bundle


def load_or_create_bundle():
    if not os.path.exists(MODEL_PATH):
        return train_and_save_pipeline()

    bundle = load(MODEL_PATH)
    required_charts = [
        os.path.join(CHART_DIR, "real_vs_predicho.png"),
        os.path.join(CHART_DIR, "residuos.png"),
    ]
    if not all(os.path.exists(path) for path in required_charts):
        return train_and_save_pipeline()
    return bundle


BUNDLE = load_or_create_bundle()
PIPELINE = BUNDLE["pipeline"]
METRICS = BUNDLE["metrics"]


def default_form_values():
    df = load_dataset()
    median = df[FEATURES].median(numeric_only=True)
    return {
        "age": int(round(median["age"])),
        "sex": "0",
        "hnr": round(float(median["hnr"]), 3),
        "dfa": round(float(median["dfa"]), 3),
        "ppe": round(float(median["ppe"]), 3),
        "rpde": round(float(median["rpde"]), 3),
    }


def parse_form(form):
    errors = []
    values = {}

    sex = form.get("sex", "0")
    if sex not in {"0", "1"}:
        errors.append("Selecciona un sexo valido.")
    values["sex"] = int(sex) if sex in {"0", "1"} else 0

    for field, info in FIELD_INFO.items():
        raw_value = form.get(field, "").strip()
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{info['label']} debe ser un numero.")
            values[field] = raw_value
            continue

        if value < info["min"] or value > info["max"]:
            errors.append(
                f"{info['label']} debe estar entre {info['min']} y {info['max']}."
            )
        values[field] = value

    return values, errors


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    errors = []
    form_values = default_form_values()

    if request.method == "POST":
        form_values, errors = parse_form(request.form)
        if not errors:
            input_df = pd.DataFrame([{field: form_values[field] for field in FEATURES}])
            prediction = float(PIPELINE.predict(input_df)[0])

    return render_template(
        "index.html",
        metrics=METRICS,
        field_info=FIELD_INFO,
        form_values=form_values,
        prediction=prediction,
        errors=errors,
        features=FEATURES,
    )


@app.route("/descargar-metricas")
def descargar_metricas():
    df = pd.DataFrame(
        [
            {"Metrica": "R2 promedio CV=7", "Valor": METRICS["cv_r2"]},
            {"Metrica": "MAE promedio CV=7", "Valor": METRICS["cv_mae"]},
            {"Metrica": "Desviacion R2 CV=7", "Valor": METRICS["cv_r2_std"]},
            {"Metrica": "Varianza R2 CV=7", "Valor": METRICS["cv_r2_var"]},
            {"Metrica": "R2 prueba 80/20", "Valor": METRICS["split_r2"]},
            {"Metrica": "MAE prueba 80/20", "Valor": METRICS["split_mae"]},
        ]
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Metricas", index=False)
        pd.DataFrame({"Caracteristicas": FEATURES}).to_excel(
            writer, sheet_name="Variables", index=False
        )
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="metricas_parkinsons.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    app.run(debug=False, port=5055, use_reloader=False)
