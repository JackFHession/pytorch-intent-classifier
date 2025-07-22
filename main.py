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

print("Jarvis is online!")
while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    tokens = tokenize(sentence)
    print(tokens)
    X = bag_of_words(tokens, all_words)
    print(X)
    X = torch.from_numpy(X).float().unsqueeze(0)
    print(X)

    output = model(X)
    print(output)
    _, predicted = torch.max(output, dim=1)
    print(_)
    print(predicted)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print("Jarvis:", random.choice(intent["responses"]))
    else:
        print("EURI: Sorry, I didn't understand.")