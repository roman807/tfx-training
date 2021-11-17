import os
from datetime import datetime

from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig

from laptop_pipeline import create_pipeline


_laptop_root = os.path.join(os.environ['HOME'], 'laptop')
_pipeline_name = 'laptop_pipeline_airflow'
_data_root = os.path.join(_laptop_root, 'data')
_trainer_module_file = os.path.join(_laptop_root, 'utils/trainer_module.py')
_transform_module_file = os.path.join(_laptop_root, 'utils/transform_module.py')
_serving_model_dir = os.path.join(_laptop_root, 'serving_model', _pipeline_name)
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime(2019, 1, 1),
}


DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
    create_pipeline(
        data_root=_data_root,
        transform_module_file=_transform_module_file,
        trainer_module_file=_trainer_module_file,
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        serving_model_dir=_serving_model_dir,
        metadata_path=_metadata_path,
    )
)
