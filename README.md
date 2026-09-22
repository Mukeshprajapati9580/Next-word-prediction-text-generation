# 🧠 RNN / LSTM Next Word Predictor

A Streamlit-based text generation application that uses a trained LSTM (Long Short-Term Memory) model to predict the next word and generate text from user-provided seed text.

## 📌 Project Overview

This project trains an RNN/LSTM language model and provides a simple Streamlit interface for text generation.

The application:
- Accepts seed text from the user.
- Predicts the next word using the trained LSTM model.
- Generates multiple words iteratively.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- Pickle
- LSTM / RNN
- Tokenization
- Sequence Padding

## 📂 Project Structure

RNN-Text-Generator/
│
├── app.py
├── Code.ipynb
├── lstm_model.h5
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
├── README.md
└── .gitignore

## 🧠 Model Details

The trained model uses:

- Model type: LSTM
- Embedding dimension: 50
- LSTM/RNN units: 128
- Output layer: Dense with softmax activation
- Optimizer: Adam
- Loss function: Categorical Crossentropy

The notebook uses a tokenizer to convert text into sequences and `pad_sequences` to make sequences a fixed length. The LSTM model then predicts the next word.

## ⚙️ Installation

### 1. Clone the repository

git clone (https://github.com/Mukeshprajapati9580/Next-word-prediction-text-generation)

### 2. Create a virtual environment (recommended)

Windows:

python -m venv venv
venv\Scripts\activate


### 3. Install dependencies

pip install -r requirements.txt

## ▶️ Run the Streamlit Application

Make sure these files are in the same directory as `app.py`:

lstm_model.h5
tokenizer.pkl
max_len.pkl

Then run:

python -m streamlit run app.py

The Streamlit application will open in your browser.

## 💻 How to Use

1. Enter seed text in the **Enter seed text** box.
2. Select the number of words to generate.
3. Click **✨ Generate Text**.
4. The application displays the generated text.
5. The app also shows the next predicted word when it can recognize the input.
