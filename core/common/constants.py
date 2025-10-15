META_DATA = 'metadata'
OUTPUT = 'output'
TOTAL_LOSS = 'total-loss'
LOCAL_LOSS = 'local-loss'
PERPLEXITY = 'perplexity'
OBSERVATION = 'observation'
BATCH_SIZE = 'batch_size'

TRAIN = "train"
DEV = "dev"
TEST = "test"

# Musical Grammar
PITCH_NAME_LABELS = ['C', 'C#(Db)', 'D', 'D#(Eb)', 'E', 'F', 'F#(Gb)', 'G', 'G#(Ab)', 'A', 'A#(Bb)', 'B']
PITCH_NAME_LABELS_PRIOR_DICT = {
    'M': ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'],
    'm': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
}
SHARP_CIRCLE = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#']
FLAT_CIRCLE = ['C', 'F', 'B-', 'E-', 'A-', 'D-', 'G-', 'C-']

# Dataset
KEY_PREPROCESS_NONE = 'key-preprocess-none'
KEY_PREPROCESS_NORMALIZE = 'key-preprocess-normalized'
KEY_PREPROCESS_AUGMENTED = 'key-preprocess-augmented'

REST_INDEX = -1  # must be -1
SPECIAL_INDEX = -2
PAD_INDEX = -3
MASK_ON_INDEX = 1
UNK_STR = '<unk>'
PAD_STR = '<pad>'
DURATIONS_SPECIAL_SYMBOL = 1.0  # dummy duration for special symbols
DURATIONS_PAD = 0.0
METRIC_BEAT_RATIO = 0.25
BEAT_SPECIAL_SYMBOL = 0.0
MEASURE_SPECIAL_SYMBOL = -1
RESID_STATE = -1
PITCHCLASS = 'pitchclass'
MIDI = 'midi'

LOG_BEST_JOINT_PROB = 'logbestprob'
NLL = 'nll'
STATES = 'states'
RESIDENCES = 'residences'
BEAT_POSITIONS = 'beat-positions'
