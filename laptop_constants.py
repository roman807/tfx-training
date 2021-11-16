MAX_VOCAB_LEN = 25
NUMERIC_FEATURE_KEYS = ['Inches', 'ScreenResolution', 'Weight']
VOCAB_FEATURE_DICT = {
    'Company': 19,
    'Product': MAX_VOCAB_LEN,
    'TypeName': 6,
    'Cpu': MAX_VOCAB_LEN,
    'Ram': 9,
    'Memory': MAX_VOCAB_LEN,
    'Gpu': MAX_VOCAB_LEN,
    'OpSys': 9,
}
NUM_OOV_BUCKETS = 3
LABEL_KEY = 'Price_euros'