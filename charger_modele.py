
import mlflow
import mlflow.sklearn
import pandas as pd


# ===== 1. Retrouver le meilleur run =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

if runs.empty:
    raise ValueError("Aucun run trouvé dans l'expérience churn-prediction.")

run_id = runs.iloc[0]["run_id"]


# ===== 2. Recharger le meilleur modèle =====
model = mlflow.sklearn.load_model(
    f"runs:/{run_id}/model"
)

print("Modèle rechargé :", type(model).__name__)
print("Nombre de variables attendues :", model.n_features_in_)


# ===== 3. Créer un nouveau client =====
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,
}])


# ===== 4. Adapter les données aux variables du modèle =====
client = client.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)


# ===== 5. Prédire le départ du client =====
prediction = model.predict(client)[0]

print(
    "Le client va-t-il partir ? ->",
    "OUI" if prediction == 1 else "NON"
)
