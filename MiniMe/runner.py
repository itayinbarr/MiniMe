from back import *
from preprocess import *
# ---------------------------

# Introduction
print("***************************")
print("***************************")
print("Welcome to MiniMe - A recurrent neural network model I trained, to challenge people to decide who wrote a sequence of words: the AI, or me?")
print("---------------------------")
print("---------------------------")

# Run once and comment out
# text_preprocess('friends_whatsapp_chat.txt')
# create_model()

challenge = input("Type down a start for the sentence")
use_model(challenge)
