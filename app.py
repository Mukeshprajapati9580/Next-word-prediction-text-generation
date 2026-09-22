import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="RNN Text Generator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 RNN / LSTM Next Word Predictor")
st.write("Enter some text and the trained LSTM model will generate the next words.")

@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


try:
    model, tokenizer, max_len = load_resources()
except Exception as e:
    st.error("Could not load the trained model or supporting files.")
    st.info(
        "Keep these files in the same folder as app.py: "
        "lstm_model.h5, tokenizer.pkl, and max_len.pkl"
    )
    st.stop()


def predict_next_word(text):
    text = text.lower().strip()

    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return None

    padded_sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    prediction = model.predict(padded_sequence, verbose=0)
    predicted_index = int(np.argmax(prediction, axis=1)[0])

    return tokenizer.index_word.get(predicted_index)


def generate_text(seed_text, n_words):
    generated_text = seed_text.strip()

    for _ in range(n_words):
        next_word = predict_next_word(generated_text)

        if not next_word:
            break

        generated_text += " " + next_word

    return generated_text


st.subheader("Text Generation")

seed_text = st.text_area(
    "Enter seed text",
    value="the meaning of life",
    height=100,
    placeholder="Example: the meaning of life"
)

n_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=30,
    value=10
)

if st.button("✨ Generate Text", use_container_width=True):
    if not seed_text.strip():
        st.warning("Please enter some seed text.")
    else:
        with st.spinner("Generating text..."):
            result = generate_text(seed_text, n_words)

        st.subheader("Generated Text")
        st.success(result)

        next_word = predict_next_word(seed_text)
        if next_word:
            st.write(f"**Next predicted word:** `{next_word}`")
        else:
            st.warning(
                "The model could not recognize any word from the input."
            )

with st.expander("Model Information"):
    st.write(f"**Model:** LSTM")
    st.write(f"**Vocabulary size:** {len(tokenizer.word_index)}")
    st.write(f"**Maximum sequence length:** {max_len}")
    st.write(f"**Embedding dimension:** 50")
    st.write(f"**RNN/LSTM units:** 128")