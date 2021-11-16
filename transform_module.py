import tensorflow as tf
import tensorflow_transform as tft

# import constants from cells above
import laptop_constants

# Unpack the contents of the constants module
# _MAX_VOCAB_LEN = laptop_constants.MAX_VOCAB_LEN
_NUMERIC_FEATURE_KEYS = laptop_constants.NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_DICT = laptop_constants.VOCAB_FEATURE_DICT
_NUM_OOV_BUCKETS = laptop_constants.NUM_OOV_BUCKETS
_LABEL_KEY = laptop_constants.LABEL_KEY


def _screen_res_width(screen_res):
    screen_res = tf.strings.split(screen_res, "x")
    return tf.reshape(tf.strings.regex_replace(screen_res[0], '[^0-9.]', ''), (1,))


# Define the transformations
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """

    # Initialize outputs dictionary
    outputs = {}

    weight = tf.strings.to_number(
        tf.map_fn(
            lambda x: tf.strings.regex_replace(x, "[aA-zZ]", ""),
            inputs['Weight']),
        out_type=tf.dtypes.float32)

    screen_resolution = tf.strings.to_number(
        tf.map_fn(
            _screen_res_width,
            tf.squeeze(inputs['ScreenResolution'], axis=1)),
        out_type=tf.dtypes.float32)

    inches = inputs['Inches']

    numeric_features_preprocessed = {
        'Inches': inches,
        'ScreenResolution': screen_resolution,
        'Weight': weight
    }

    for key, value in numeric_features_preprocessed.items():  # _NUMERIC_FEATURE_KEYS:
        scaled = tft.scale_to_0_1(value)
        outputs[key] = tf.reshape(scaled, [-1])

    # Convert strings to indices and convert to one-hot vectors
    for key, vocab_size in _VOCAB_FEATURE_DICT.items():
        indices = tft.compute_and_apply_vocabulary(inputs[key], num_oov_buckets=_NUM_OOV_BUCKETS)
        one_hot = tf.one_hot(indices, vocab_size + _NUM_OOV_BUCKETS)
        outputs[key] = tf.reshape(one_hot, [-1, vocab_size + _NUM_OOV_BUCKETS])

    # Cast label to float
    outputs[_LABEL_KEY] = tf.cast(inputs[_LABEL_KEY], tf.float32)

    return outputs