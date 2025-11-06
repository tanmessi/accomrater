import sys
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam

# Project imports (copied/adapted from Hotel-v1.py)
from processors.vlsp2018_processor import VLSP2018Loader
from processors.vietnamese_processor import VietnameseTextPreprocessor
from transformers import AutoTokenizer
from acsa_model import VLSP2018MultiTask

# Basic constants (match Hotel-v1.py)
TRAIN_PATH = r'./datasets/vlsp2018_hotel/1-VLSP2018-SA-Hotel-train.csv'
VAL_PATH = r'./datasets/vlsp2018_hotel/2-VLSP2018-SA-Hotel-dev.csv'
TEST_PATH = r'./datasets/vlsp2018_hotel/3-VLSP2018-SA-Hotel-test.csv'
PRETRAINED_MODEL = 'vinai/phobert-base'
MODEL_NAME = 'Hotel-v1'
MAX_LENGTH = 256

# Load dataset metadata to obtain aspect/category names
raw_datasets = VLSP2018Loader.load(TRAIN_PATH, VAL_PATH, TEST_PATH)
ASPECT_CATEGORY_NAMES = raw_datasets['train'].column_names[1:]

# Tokenizer and Vietnamese preprocessor (copied mapping from Hotel-v1.py)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
vn_preprocessor = VietnameseTextPreprocessor(vncorenlp_dir='processors/VnCoreNLP', extra_teencodes={
    'khách sạn': ['ks'], 'nhà hàng': ['nhahang'], 'nhân viên': ['nv'],
    'cửa hàng': ['store', 'sop', 'shopE', 'shop'],
    'sản phẩm': ['sp', 'product'], 'hàng': ['hàg'],
    'giao hàng': ['ship', 'delivery', 'síp'], 'đặt hàng': ['order'],
    'chuẩn chính hãng': ['authentic', 'aut', 'auth'], 'hạn sử dụng': ['date', 'hsd'],
    'điện thoại': ['dt'],  'facebook': ['fb', 'face'],
    'nhắn tin': ['nt', 'ib'], 'trả lời': ['tl', 'trl', 'rep'],
    'feedback': ['fback', 'fedback'], 'sử dụng': ['sd'], 'xài': ['sài'],
}, max_correction_length=MAX_LENGTH)

# Instantiate model and load weights. Use a simple Adam optimizer here so the model can be constructed.
optimizer = Adam(learning_rate=1e-4)
reloaded_model = VLSP2018MultiTask(PRETRAINED_MODEL, ASPECT_CATEGORY_NAMES, optimizer, name=MODEL_NAME)
try:
    reloaded_model.load_weights(f'./weights/{MODEL_NAME}/{MODEL_NAME}')
    print(f'Loaded weights from ./weights/{MODEL_NAME}/{MODEL_NAME}')
except Exception as e:
    print('Warning: could not load model weights:', e, file=sys.stderr)
    print('You can still run preprocessing/tokenization but predictions will likely fail until weights are available.', file=sys.stderr)

# Interactive prediction (same as in Hotel-v1.py)
random_input = VLSP2018Loader.preprocess_and_tokenize(
    input('Enter your review: '), vn_preprocessor, tokenizer,
    batch_size=1, max_length=MAX_LENGTH
)
tf_inputs = Dataset.from_tensor_slices({x: [[random_input[x][0]]] for x in tokenizer.model_input_names})
random_pred = reloaded_model.acsa_predict(tf_inputs)
reloaded_model.print_acsa_pred(random_pred[0])