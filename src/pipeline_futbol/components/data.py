from kfp.dsl import Dataset, Output, component

# ##########################################
# DATA SPLIT
# ##########################################

@component(
    base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-14.py310:latest",
    packages_to_install=[
        "pandas>=1.5.0,<2.1.4",
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

    # La columna 'date' viene en formato yyyy-mm-dd como string; para evitar el
    # error de sklearn al intentar convertirla a float, la convertimos a un valor
    # numérico representativo (ordinal del día).
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df["date"] = df["date"].map(pd.Timestamp.toordinal)

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
