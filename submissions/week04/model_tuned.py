import os
import sys
import random
import string
import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')    
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras import layers, callbacks


MODEL_BASENAME = "submissions/week04/model_tuned"
MODEL_EXT      = ".keras"

def build_and_train_model(x_train, y_train, x_val, y_val,
                          max_len_x, len_chars_in,
                          max_len_y, len_chars_out,
                          latent_dim=256, batch_size=64, epochs=50):
    model = Sequential([
        layers.Input(shape=(max_len_x, len_chars_in)),
        layers.LSTM(latent_dim, return_sequences=False),
        layers.RepeatVector(max_len_y),
        layers.LSTM(latent_dim, return_sequences=True),
        layers.TimeDistributed(layers.Dense(len_chars_out)),
        layers.Activation('softmax')
    ])
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    print("\nModel Architecture Summary:")
    model.summary()

    cb = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
    ]
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=cb
    )

    # plot & save training curves
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("submissions/week04/training_plot.png")
    plt.close()
    print("Training plot saved to submissions/week04/training_plot.png\n")

    return model

def main():
    # -- 1) Morse dictionaries & conversion
    alphabet = list('abcdefghijklmnopqrstuvwxyz-')
    values   = [
        '.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..',
        '.---', '-.-', '.-..', '--', '-.', '---', '.--.', '--.-', '.-.',
        '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..', '-....-'
    ]
    morse_dict = dict(zip(alphabet, values))
    ascii_dict = {v: k for k, v in morse_dict.items()}

    def morse_encode(text):
        t = ''.join(c for c in text.lower() if c in alphabet)
        return ' '.join(morse_dict[c] for c in t)

    def morse_decode(code):
        return ''.join(ascii_dict[c] for c in code.split(' '))

    # -- 2) Embedding helper
    class Embedding:
        def __init__(self, chars):
            self.chars = sorted(set(chars))
            self.char_to_idx = {c:i for i,c in enumerate(self.chars)}
            self.idx_to_char = {i:c for i,c in enumerate(self.chars)}
        def encode(self, token, num_rows):
            X = np.zeros((num_rows, len(self.chars)), dtype=float)
            for i, c in enumerate(token):
                X[i, self.char_to_idx[c]] = 1.0
            return X
        def decode(self, X):
            # X is one-hot vectors stacked
            idx = X.argmax(axis=-1)
            return ''.join(self.idx_to_char[i] for i in idx)

    # -- 3) Data prep
    word_len = 6
    max_len_x = max(len(v) for v in values)*word_len + (word_len-1)
    max_len_y = word_len

    def gen_data(n):
        outs = [
            ''.join(random.choice(string.ascii_lowercase + '-') 
                    for _ in range(word_len))
            for _ in range(n)
        ]
        ins = [morse_encode(s) for s in outs]
        return outs, ins

    outs, ins = gen_data(10000)
    chars_in  = '-. '
    chars_out = ''.join(alphabet)

    emb_in  = Embedding(chars_in)
    emb_out = Embedding(chars_out)

    X = np.zeros((len(ins),  max_len_x, len(chars_in)),  dtype=float)
    Y = np.zeros((len(outs), max_len_y, len(chars_out)), dtype=float)

    for i, token in enumerate(ins):
        X[i] = emb_in.encode(token, max_len_x)
    for i, token in enumerate(outs):
        Y[i] = emb_out.encode(token, max_len_y)

    split = 3*len(X)//4
    x_train, x_val = X[:split], X[split:]
    y_train, y_val = Y[:split], Y[split:]

    # -- 4) Model load / retrain logic
    model_path = MODEL_BASENAME + MODEL_EXT

    if os.path.exists(model_path):
        print(f"Found existing model file: '{model_path}'.")
        ans = input("[L]oad & test OR [R]etrain & save new? (L/R): ").strip().lower()
        if ans == 'l':
            model = load_model(model_path)
            print("\nLoaded successfully. Architecture summary:")
            model.summary()
        elif ans == 'r':
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_path = f"{MODEL_BASENAME}_{ts}{MODEL_EXT}"
            print(f"\nRetraining and will save to '{new_path}'\n")
            model = build_and_train_model(
                x_train, y_train, x_val, y_val,
                max_len_x, len(chars_in),
                max_len_y, len(chars_out),
                latent_dim=256, batch_size=64, epochs=45
            )
            model.save(new_path)
            print(f"New model saved to '{new_path}'")
        else:
            print("Invalid choice, defaulting to load.")
            model = load_model(model_path)

    else:
        print("No saved model found. Training a new model...\n")
        model = build_and_train_model(
            x_train, y_train, x_val, y_val,
            max_len_x, len(chars_in),
            max_len_y, len(chars_out),
            latent_dim=256, batch_size=64, epochs=45
        )
        model.save(model_path)
        print(f"Model saved to '{model_path}'")

    # -- 5) Testing on some examples
    tests = ['abcdef', '-hslu-', 'hahaha', 'tttuuu', 'sos']
    T = np.zeros((len(tests), max_len_x, len(chars_in)), dtype=float)
    for i, tok in enumerate(tests):
        T[i] = emb_in.encode(morse_encode(tok), max_len_x)

    preds = model.predict(T)
    print("\n--- TEST RESULTS ---")
    for i, tok in enumerate(tests):
        inp  = emb_in.decode(T[i])
        outp = emb_out.decode(preds[i])
        print(f"{inp} → {outp}   <->   {tok}")

if __name__ == "__main__":
    main()
