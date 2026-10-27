
import mlflow

# ===== 1. Récupérer les runs de l'expérience =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# ===== 2. Afficher les résultats =====
colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

print(runs[colonnes].to_string(index=False))

# ===== 3. Identifier le meilleur run =====
print()

if not runs.empty:
    meilleur = runs.iloc[0]

    print(
        f"Meilleur run : {meilleur['tags.mlflow.runName']} "
        f"| id : {meilleur['run_id'][:8]}..."
    )
else:
    print("Aucun run trouvé dans l'expérience churn-prediction.")
