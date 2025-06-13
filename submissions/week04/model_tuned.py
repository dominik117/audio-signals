import random
import string
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras import layers, callbacks
from keras.utils import plot_model
import os

# -----------------------------------------------------------------------------
# Morse Code Dictionaries
alphabet = list('abcdefghijklmnopqrstuvwxyz-')
values = [
    '.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..', '.---', '-.-',
    '.-..', '--', '-.','---', '.--.', '--.-', '.-.',
    '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..','-....-'
]
morse_dict = dict(zip(alphabet, values))
ascii_dict = dict(map(reversed, morse_dict.items()))

# -----------------------------------------------------------------------------
# Morse Conversion Functions
def morse_encode(text):
    t = ''.join([c for c in text.lower() if c in alphabet])
    return ' '.join([''.join(morse_dict[i]) for i in t])

def morse_decode(code):
    return ''.join([ascii_dict[i] for i in code.split(' ')])

# -----------------------------------------------------------------------------
# Embedding Helper Class
class Embedding:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, token, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(token):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x):
        x = [x.argmax(axis=-1)]
        return ''.join(self.indices_char[int(v)] for v in x)

# -----------------------------------------------------------------------------
# Data Preparation
word_len = 6
max_len_x = len(max(values, key=len)) * word_len + (word_len - 1)
max_len_y = word_len

def generate_data(n):
    output_list = [''.join(random.choice(string.ascii_lowercase + '-') for _ in range(word_len)) for _ in range(n)]
    return output_list, [morse_encode(s) for s in output_list]

output_list, input_list = generate_data(10000)
chars_in = '-. '
chars_out = ''.join(alphabet)

embedding_in = Embedding(chars_in)
embedding_out = Embedding(chars_out)

x = np.zeros((len(input_list), max_len_x, len(chars_in)))
y = np.zeros((len(output_list), max_len_y, len(chars_out)))

for i, token in enumerate(input_list):
    x[i] = embedding_in.encode(token, max_len_x)
for i, token in enumerate(output_list):
    y[i] = embedding_out.encode(token, max_len_y)

m = 3 * len(x) // 4
x_train, x_val = x[:m], x[m:]
y_train, y_val = y[:m], y[m:]

# -----------------------------------------------------------------------------
# Model Configuration
latent_dim = 256
Batch_size = 64
Epochs = 75
model_path = "model_4-1_tuned.h5"

model = Sequential([
    layers.LSTM(latent_dim, input_shape=(max_len_x, len(chars_in)), return_sequences=False),
    layers.RepeatVector(max_len_y),
    layers.LSTM(latent_dim, return_sequences=True),
    layers.TimeDistributed(layers.Dense(len(chars_out))),
    layers.Activation('softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Optional: visualize model structure
if not os.path.exists("model_plot.png"):
    plot_model(model, to_file="model_plot.png", show_shapes=True, dpi=100)

# -----------------------------------------------------------------------------
# Training
cb = [callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
hist = model.fit(
    x_train, y_train,
    batch_size=Batch_size,
    epochs=Epochs,
    validation_data=(x_val, y_val),
    callbacks=cb
)

# -----------------------------------------------------------------------------
# Show Progress
def show_progress():
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

show_progress()

# -----------------------------------------------------------------------------
# Testing on Known Data
test = ['abcdef', '-hslu-', 'hahaha', '------', 'tttuuu']
test_x = np.zeros((len(test), max_len_x, len(chars_in)))

for i, token in enumerate(test):
    test_x[i] = embedding_in.encode(morse_encode(token), max_len_x)

preds = model.predict(test_x)

for i in range(len(test)):
    morse_seq = ''.join([embedding_in.decode(frame) for frame in test_x[i]])
    decoded = ''.join([embedding_out.decode(frame) for frame in preds[i]])
    print(f"{morse_seq} → {decoded}  <->  {test[i]}")

# -----------------------------------------------------------------------------
# Save Model
model.save(model_path)
print(f"Model saved to {model_path}")
