import random
import torch
from ai.model import NeuralNet
from ai.utils import bag_of_words, tokenize
import json

with open("datasets/intents.json", 'r') as f:
    intents = json.load(f)

data = torch.load("trained_model.pth")

model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags = data["tags"]

print("Jarvis is online! Type 'quit' to exit.")
while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    tokens = tokenize(sentence)
    X = bag_of_words(tokens, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print("Jarvis:", random.choice(intent["responses"]))
    else:
        print("EURI: Sorry, I didn't understand.")