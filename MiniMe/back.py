import tensorflow as tf
import os
import numpy as np


# Used once to create and initially train the model
# with the messages I sent to my friend Tamir
def create_model():
    # Open the processed file which contains initial dataset
    text = open('data/tamirherzberg - processed.txt', 'rb').read().decode(encoding='utf-8')

    # Find all unique characters
    vocab = sorted(set(text))

    # Create a character index for turning input to numbers.
    char2idx = {u: i for i, u in enumerate(vocab)}

    # Turning character to number with char2idx
    def text_to_int(text):
        return np.array([char2idx[c] for c in text])

    text_as_int = text_to_int(text)

    # Setting length of each training sample
    seq_length = 100
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    # Creating input and output
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    # Sizes of RNN features
    BATCH_SIZE = 64
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 256
    RNN_UNITS = 1024
    BUFFER_SIZE = 10000

    # Using decided batch size
    data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Building the model based on decided sizes
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

    # Used loss function
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    # Compiling model
    model.compile(optimizer='adam', loss=loss)

    # saving each epoch checkpoint
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    # Fitting the model
    model.fit(data, epochs=50, callbacks=[checkpoint_callback])


# Building the RNN model.
# Used again to recreate a model with the specific input and output size
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


# Using the model to predict an output
def use_model(input_text):
    # Recreating character index
    text = open('data/tamirherzberg - processed.txt', 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    # Turning input text to numbers
    input_eval = [char2idx[s] for s in input_text]
    input_eval = tf.expand_dims(input_eval, 0)

    # Determining sizes of RNN model
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 256
    RNN_UNITS = 1024
    temperature = 1.0

    # Creating model with batch size fit to predict based on one input
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

    # Loading the trained weights of the model
    checkpoint_dir = './training_checkpoints'
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    # Number of characters to predict
    num_generate = 15
    text_generated = []

    # Resetting the last state the weights were in
    model.reset_states()

    # Predicting the output
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    # Printing output
    print(input_text + ''.join(text_generated))
