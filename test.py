import torch.nn.functional as F
import torch.nn as nn
import torch
import openai

from src.gpt import chatgpt

prompt = '''Giving 4 plans to divide the following labels into 2 categories.
Input: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
'''
client = openai.OpenAI()
res = chatgpt(client, prompt, model='gpt-3.5-turbo', temperature=0.7, max_tokens=1000, n=1, stop=None)
print(res)

'''Giving 4 plans to divide the following labels into 2 categories.
Input: 'motor vehicle', 'craft', 'placental', 'bird'
Answer: plan1
'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
'''

'''
Give me 4 plans to divide the following 10 labels into 2 categories and give the reasons for the classification, each word and it's definition are of the form word(definition):
airplane(A powered heavier-than air aircraft with fixed wings.)
automobile(An enclosed passenger vehicle powered by an engine.)
bird(A member of the class of animals Aves in the phylum Chordata, characterized by being warm-blooded, having feathers and wings usually capable of flight, and laying eggs.)
cat(A domesticated species of feline animal, commonly kept as a house pet.)
deer(a ruminant mammal with antlers and hooves of the family Cervidae or one of several similar animals from related families of the order Artiodactyla.)
dog(An animal, member of the genus Canis (probably descended from the common wolf) that has been domesticated for thousands of years; occurs in many breeds. Scientific name: Canis lupus familiaris.)
frog(A small hopping amphibian.)
horse(Any current or extinct animal of the family Equidae, including the zebra or the ass.)
ship(A water-borne vessel larger than a boat.)
truck(Any motor vehicle designed for carrying cargo, including delivery vans, pickups, and other motorized vehicles (including passenger autos) fitted with a bed designed to carry goods.)

Give me 4 plans to divide the following 10 words into two categories and give the reasons for the classification:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
'''

