# gcp-pipeline-models

## Arquitectura
## Setup

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
