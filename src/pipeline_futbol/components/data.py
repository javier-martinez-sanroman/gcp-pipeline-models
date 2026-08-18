from kfp.dsl import Dataset, Output, component

# ##########################################
# DATA SPLIT
# ##########################################

@component(
    # base_image="gcr.io/deeplearning-platform-release/tf2-cpu.2-6:latest",
    base_image='python:3.11',
    packages_to_install=[
        "pandas",
        "google-cloud-bigquery",
    ],
)
def load_data(
    project_id: str,
    bq_dataset: str,
    bq_table: str,
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import pandas as pd
    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split

    client = bigquery.Client()

    dataset_ref = bigquery.DatasetReference(project_id, bq_dataset)
    table_ref   = dataset_ref.table(bq_table)
    table       = bigquery.Table(table_ref)
    iterable_table = client.list_rows(table).to_dataframe_iterable()

    dfs = []
    for row in iterable_table:
        dfs.append(row)

    df = pd.concat(dfs, ignore_index=True)
    del dfs

    # # Añadir columna 'quiniela' según el resultado del partido
    # import numpy as np

    # # Asegurar que los goles son numéricos
    # df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    # df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # conditions = [
    #     df["home_score"] > df["away_score"],
    #     df["home_score"] < df["away_score"],
    #     df["home_score"] == df["away_score"],
    # ]
    # choices = ["1", "2", "X"]

    # df["quiniela"] = np.select(conditions, choices, default="X")

    # df["neutral"].replace(
    #     {
    #         "FALSE":0,
    #         "TRUE":1,
    #     },
    #     inplace=True,
    # )

    # print(df)

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop("quiniela", axis=1),
        df["quiniela"],
        test_size=0.2,
        random_state=42,
    )

    x_train["quiniela"] = y_train
    x_test["quiniela"] = y_test

    x_train.to_csv(f"{train_dataset.path}", index=False)
    x_test.to_csv(f"{test_dataset.path}", index=False)
