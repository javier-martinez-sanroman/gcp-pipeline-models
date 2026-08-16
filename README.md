# gcp-pipeline-models

## Arquitectura
## Setup project

```bash
mkdir -p gcp-pipeline-models && cd gcp-pipeline-models
```

```bash
python3 -m venv venv-gcp-pipeline-models
source venv-gcp-pipeline-models/bin/activate
# pip install --upgrade setuptools==69.5.1
pip install pre-commit
pip install -r requirements.txt
```

```bash
pre-commit run --all-files
```

## Setup GCP
[Consola Google Cloud Platform](https://console.cloud.google.com/)

* [workload identity federation](https://console.cloud.google.com/iam-admin/workload-identity-pools)
Creamos una federacion para accesos externos y agregamos un proveedor para github

    **Attribute mapping**

    - google.subject == assertion.sub
    - attribute.actor == assertion.actor
    - attribute.aud   == assertion.aud
    - attribure.repository == assertion.repository

    **Attribute conditions**

    - assertion.repository=='{github-account}/gcp-pipeline-models'

* [Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts)
Creamos una cuenta de servicio para la conexión


* [Artifact Registry](https://console.cloud.google.com/artifacts)
Creamos una cuenta de servicio para la conexión

    - artifact-repository: Kubeflow Pipelines
