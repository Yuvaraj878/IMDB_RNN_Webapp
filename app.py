import streamlit as st

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1) Load tokenizer
with open('tokenizer.json','r') as f:
    tokenizer = tokenizer_from_json(f.read())

# 2) Hyper-params
MAX_VOCAB = 10_000
MAX_LEN   = 200
EMB_DIM   = 128

# 3) Rebuild model
model = Sequential([
    Embedding(input_dim=MAX_VOCAB, output_dim=EMB_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# ── BUILD & LOAD WEIGHTS ─────────────────────────────────────────────────────────
model.build(input_shape=(None, MAX_LEN))              # ensure the model is built
model.load_weights('sentiment_rnn1.h5')            # weights-only file from training

# 4) Streamlit UI
st.title("IMDb Sentiment Analysis")
review = st.text_area("Enter a movie review:")

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        seq    = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        score  = float(model.predict(padded)[0][0])
        label  = "Positive 😊" if score > 0.5 else "Negative 😞"
        st.success(f"{label}  (confidence: {score:.3f})")
