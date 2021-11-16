import os

# import tfx
from tfx import v1 as tfx

# import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Tuner
from tfx.components import Transform
from tfx.components.trainer.executor import Executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
# from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
# from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import example_gen_pb2
import tensorflow as tf
import tensorflow_transform as tft
# import tensorflow_addons as tfa
# from tfx.orchestration import LocalDagRunner
import pandas as pd
from time import time
import numpy as np
import tensorflow_model_analysis as tfma
from google.protobuf import text_format


def _create_pipeline(
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
        # hyperparameters=tuner.outputs['best_hyperparameters'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval']))

    model_resolver = tfx_v1.dsl.Resolver(
        strategy_class=tfx_v1.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx_v1.dsl.Channel(type=tfx_v1.types.standard_artifacts.Model),
        model_blessing=tfx_v1.dsl.Channel(
            type=tfx_v1.types.standard_artifacts.ModelBlessing)).with_id(
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

    evaluator = tfx.components.Evaluator(
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
        evaluator,
        pusher
    ]

    return pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
        components=components)
