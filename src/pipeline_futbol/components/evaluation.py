from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

# ##########################################
# EVALUATION
# ##########################################
@component(
    base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-14.py310:latest",
    packages_to_install=[
        "pandas>=1.5.0,<2.1.4",
        "joblib==1.2.0",
    ],
)
def choose_best_model(
    test_dataset: Input[Dataset],
    decision_tree_model: Input[Model],
    random_forest_model: Input[Model],
    metrics: Output[Metrics],
    best_model: Output[Model],
):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score

    test_data = pd.read_csv(test_dataset.path)

    dt = joblib.load(decision_tree_model.path)
    rf = joblib.load(random_forest_model.path)

    dt_pred = dt.predict(test_data.drop("quiniela_num", axis=1))
    rf_pred = rf.predict(test_data.drop("quiniela_num", axis=1))

    df_accuracy = accuracy_score(test_data["quiniela_num"], dt_pred)
    rf_accuracy = accuracy_score(test_data["quiniela_num"], rf_pred)

    metrics.log_metric("Decision Tree (Accuracy)", (df_accuracy))
    metrics.log_metric("Random Forest (Accuracy)", (rf_accuracy))

    if df_accuracy > rf_accuracy:
        joblib.dump(dt, best_model.path)
    else:
        joblib.dump(rf, best_model.path)
