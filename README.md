MiniMe
==============================

A recurrent neural network model I created & trained, in attempt to challenge my friends to 
an advanced Turing test.

Rather than asking if a sequence of text is made up by a man or machine,
Let's wonder if the sequence is made by this model, or specifically me?

This project is a demonstration of a RNN text generator.


Getting Started
------------

From within the repo directory run

`./MiniMe/runner.py`

You can now type in the console any text you wish. This input will be the beginning of 
the sentence generated by my model

After pressing Return button, you will a get a 15 character-long text sequence.

-----
About Training & Dataset
--

This model was trained with hundreds of thousands of text messages
I sent through Whatsapp to my friends over the years.

Before using the chat, it went through a text preprocessing pipeline. 
The goal of the pipeline was to remove all English characters, numbers,
punctuation marks and unique symbols.

Because it predict each character and not words, I didn't use 
tools for NLP.

-----


Project Organization
------------

    ├── README.md                    <- The top-level README for developers using this project
    ├── LICENSE.md                   <- MIT
    ├── .gitignore                   <- For environment directories
    │
    ├── Minime                       <- Containing the software itself
    │   ├── train_checkpoints        <- Directory of trained model .gitignored
    │   ├── back.py                  <- backend code
    │   ├── preprocess.py            <- Used to preprocess test
    │   └── runner.py                <- Running the software
    │
    └── tests                        <- Tests directory, .gitignored
        └── backend_tests.py         <- Unit tests of backend
 
Dependencies
------------

- Python
- Keras
- TensorFlow
- NumPy
- WhatsApp
--------
# MiniMe
