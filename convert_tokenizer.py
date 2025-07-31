import sys, pickle
import tensorflow.keras as _tfk

sys.modules['keras']                               = _tfk
sys.modules['keras.src']                           = _tfk
sys.modules['keras.src.legacy']                    = _tfk
sys.modules['keras.src.preprocessing']             = _tfk.preprocessing
sys.modules['keras.src.legacy.preprocessing']      = _tfk.preprocessing
sys.modules['keras.src.preprocessing.text']        = _tfk.preprocessing.text
sys.modules['keras.src.legacy.preprocessing.text'] = _tfk.preprocessing.text

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)

print("✅ tokenizer.json created!")
