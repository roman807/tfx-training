from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft

import laptop_constants

_NUMERIC_FEATURE_KEYS = laptop_constants.NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_DICT = laptop_constants.VOCAB_FEATURE_DICT
_NUM_OOV_BUCKETS = laptop_constants.NUM_OOV_BUCKETS
_LABEL_KEY = laptop_constants.LABEL_KEY

N_EPOCHS = 1


def _gzip_reader_fn(filenames):
    '''Load compressed dataset

    Args:
      filenames - filenames of TFRecords to load

    Returns:
      TFRecordDataset loaded from the filenames
    '''

    # Load the dataset. Specify the compression type since it is saved as `.gz`
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=3,
              batch_size=32) -> tf.data.Dataset:
    '''Create batches of features and labels from TF Records

    Args:
      file_pattern - List of files or patterns of file paths containing Example records.
      tf_transform_output - transform output graph
      num_epochs - Integer specifying the number of times to read through the dataset.
              If None, cycles through the dataset forever.
      batch_size - An int representing the number of records to combine in a single batch.

    Returns:
      A dataset of dict elements, (or a tuple of dict elements and label).
      Each dict maps feature keys to Tensor or SparseTensor objects.
    '''
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=_LABEL_KEY)

    return dataset


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    # Get transformation graph
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        # Get pre-transform feature spec
        feature_spec = tf_transform_output.raw_feature_spec()

        # Pop label since serving inputs do not include the label
        feature_spec.pop(_LABEL_KEY)

        # Parse raw examples into a dictionary of tensors matching the feature spec
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        # Transform the raw examples using the transform graph
        transformed_features = model.tft_layer(parsed_features)

        # Get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn


def model_builder():
    '''
    Builds the model and sets up the hyperparameters to tune.

    Args:
      hp - Keras tuner object

    Returns:
      model with hyperparameters to tune
    '''

    hp_units = 20  # hp.get('units')
    hp_learning_rate = 1e-3  # hp.get('learning_rate')

    # Define input layers for numeric keys
    input_tensors_numeric = [
        tf.keras.layers.Input(name=colname, shape=(1,), dtype=tf.float32)
        for colname in _NUMERIC_FEATURE_KEYS
    ]

    # Define input layers for vocab keys
    input_tensors_categorical = [
        tf.keras.layers.Input(name=colname, shape=(vocab_size + _NUM_OOV_BUCKETS,), dtype=tf.float32)
        for colname, vocab_size in _VOCAB_FEATURE_DICT.items()
    ]

    input_tensors = input_tensors_numeric + input_tensors_categorical

    # # Concatenate numeric inputs
    # deep = tf.keras.layers.concatenate(input_numeric)
    x = tf.keras.layers.concatenate(input_tensors)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='relu')(x)
    output = tf.keras.layers.Lambda(lambda x: x * 2000.0)(x)
    model = tf.keras.Model(input_tensors, output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),  # 1e-3
                  loss=tf.keras.losses.MeanSquaredError(),
                  # loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics='mean_absolute_error',
                  # metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                  # metrics=['accuracy']
                  )

    # Print the model summary
    model.summary()

    return model


def run_fn(fn_args: FnArgs) -> None:
    """Defines and trains the model.
    Args:
      fn_args: Holds args as name/value pairs. Refer here for the complete attributes:
      https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs#attributes
    """

    # Callback for TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')  # batch  epoch

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data good for 10 epochs
    train_set = _input_fn(fn_args.train_files[0], tf_transform_output, N_EPOCHS)
    val_set = _input_fn(fn_args.eval_files[0], tf_transform_output, N_EPOCHS)

    # Load best hyperparameters
    # hp = fn_args.hyperparameters.get('values')####

    # Build the model
    model = model_builder()  # hp)####

    # Train the model
    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback],
        epochs=15  ##########
    )

    # Define default serving signature for inference requests
    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                      tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
