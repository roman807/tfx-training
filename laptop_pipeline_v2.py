import os
from datetime import datetime

from tfx import v1 as tfx
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.orchestration import metadata
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto import example_gen_pb2
import tensorflow_model_analysis as tfma
from google.protobuf import text_format


def create_pipeline(
        data_root,
        transform_module_file,
        trainer_module_file,
        serving_model_dir,
        pipeline_name,
        pipeline_root,
        metadata_path
):
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
        ]))

    example_gen = CsvExampleGen(
        input_base=data_root,
        output_config=output
    )

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module_file
    )

    trainer = Trainer(
        module_file=trainer_module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval']))

    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
        'latest_blessed_model_resolver')

    eval_config = text_format.Parse("""
      ## Model information
      model_specs {
        # This assumes a serving model with signature 'serving_default'.
        signature_name: "serving_default",
        label_key: "Price_euros"
      }

      ## Post training metric information
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        metrics {
          class_name: "MeanAbsoluteError"
          threshold {
            # Ensure that metric is always < XXX
            value_threshold {
              upper_bound { value: 300 }
            }
            # Ensure that metric does not drop by more than a small epsilon
            # e.g. (candidate - baseline) > -1e-10 or candidate > baseline - 1e-10
            change_threshold {
              direction: LOWER_IS_BETTER
              absolute { value: 1e4 }
            }
          }
        }
        metrics { class_name: "MeanSquaredError" }
        # ... add additional metrics and plots ...
      }

      ## Slicing information
      slicing_specs {}  # overall slice

    """, tfma.EvalConfig())

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
        components=components)


# ************* Directories, config and DAG-runner for runs with Apache Airflow ************* #
_pipeline_name = 'laptop_pipeline_airflow'
_laptop_root = os.path.join(os.environ['HOME'], 'laptop')
_data_root = os.path.join(_laptop_root, 'data')
_trainer_module_file = os.path.join(_laptop_root, 'trainer_module.py')
_transform_module_file = os.path.join(_laptop_root, 'transform_module.py')
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
