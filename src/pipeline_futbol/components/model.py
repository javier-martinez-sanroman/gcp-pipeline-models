from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

# ##########################################
# DECISION TREE MODEL
# ##########################################
@component(
    base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-14.py310:latest",
    packages_to_install=[
        "pandas>=1.5.0,<2.1.4",
        "joblib==1.2.0",
    ],
)
def decision_tree(
    train_dataset: Input[Dataset],
    metrics: Output[Metrics],
    output_model: Output[Model],
):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    train = pd.read_csv(train_dataset.path)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train["date"] = pd.to_datetime(train["date"], format="%Y-%m-%d", errors="coerce")
    train["date"] = train["date"].map(pd.Timestamp.toordinal)

    train['home_team_num'] = le.fit_transform(train['home_team'])
    train['away_team_num'] = le.fit_transform(train['away_team'])
    train['neutral_num'] = le.fit_transform(train['neutral'])

    train.drop(["home_team","away_team","home_score","away_score","tournament","city","country"], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        train.drop("quiniela", axis=1),
        train["quiniela"],
        test_size=0.2,
        random_state=42,
    )

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)

    metrics.log_metric("accuracy", (acc))

    joblib.dump(model, output_model.path)

# ##########################################
# RANDOM FOREST MODEL
# ##########################################
@component(
    base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-14.py310:latest",
    packages_to_install=[
        "pandas>=1.5.0,<2.1.4",
        "joblib==1.2.0",
    ],
)
def random_forest(
    train_dataset: Input[Dataset],
    metrics: Output[Metrics],
    output_model: Output[Model],
):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    train = pd.read_csv(train_dataset.path)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train["date"] = pd.to_datetime(train["date"], format="%Y-%m-%d", errors="coerce")
    train["date"] = train["date"].map(pd.Timestamp.toordinal)

    train['home_team_num'] = le.fit_transform(train['home_team'])
    train['away_team_num'] = le.fit_transform(train['away_team'])
    train['neutral_num'] = le.fit_transform(train['neutral'])

    train.drop(["home_team","away_team","home_score","away_score","tournament","city","country"], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        train.drop("quiniela", axis=1),
        train["quiniela"],
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)

    metrics.log_metric("accuracy", (acc))

    joblib.dump(model, output_model.path)
