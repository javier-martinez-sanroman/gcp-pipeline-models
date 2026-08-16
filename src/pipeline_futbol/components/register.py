from kfp.dsl import Input, Model, Output, component

# ##########################################
# EVALUATION
# ##########################################
@component(
    base_image="gcr.io/deeplearning-platform-release/tf2-cpu.2-6:latest",
    packages_to_install=["google_cloud_aiplatform"],
)
def register_model(
    project_id: str,
    location: str,
    model: Input[Model],
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    aiplatform.Model.upload_scikit_learn_model_file(
        model_file_path=model.path,
        display_name="Futbol_Model_v1",
        project=project_id,
    )
