import torch.nn.functional as F
import torch.nn as nn
import torch
import openai
import os
import numpy as np
from torchmetrics.clustering import DunnIndex

from src.gpt import GPT
import json


prompt = '''Giving 2 plans to classify all values from Input into different sets with the class name and only word id, according to the values' definition in Wordnet. Each value can only belong to one set.\
Input: {0: "airplane", 1: "bird", 2: "cat", 3: "dog"} \
Answer: {"Plan1": {"FlyingObjects": [0, 1], "Pets": [2, 3]}, "Plan2": {"NonHuman": [0, 1, 2], "Pets": [3]}} \
Input: {25: 'european_fire_salamander.n.01', 26: 'common_newt.n.01', 27: 'eft.n.01', 28: 'spotted_salamander.n.01', 29: 'axolotl.n.01', 30: 'bullfrog.n.01', 31: 'tree_frog.n.02', 32: 'tailed_frog.n.01', 33: 'loggerhead.n.02', 34: 'leatherback_turtle.n.01', 35: 'mud_turtle.n.01', 36: 'terrapin.n.01', 37: 'box_turtle.n.01', 38: 'banded_gecko.n.01', 39: 'common_iguana.n.01', 40: 'american_chameleon.n.01', 41: 'whiptail.n.01', 42: 'agama.n.01', 43: 'frilled_lizard.n.01', 44: 'alligator_lizard.n.01', 45: 'gila_monster.n.01', 46: 'green_lizard.n.01', 47: 'african_chameleon.n.01', 48: 'komodo_dragon.n.01', 49: 'african_crocodile.n.01', 50: 'american_alligator.n.01', 52: 'thunder_snake.n.01', 53: 'ringneck_snake.n.01', 54: 'hognose_snake.n.01', 55: 'green_snake.n.02', 56: 'king_snake.n.01', 57: 'garter_snake.n.01', 58: 'water_snake.n.01', 59: 'vine_snake.n.01', 60: 'night_snake.n.01', 61: 'boa_constrictor.n.01', 62: 'rock_python.n.01', 63: 'indian_cobra.n.01', 64: 'green_mamba.n.01', 65: 'sea_snake.n.01', 66: 'horned_viper.n.01', 67: 'diamondback.n.01', 68: 'sidewinder.n.01'}\
'''

client = openai.OpenAI()
messages = [
    {"role": "system", "content": 'You are a helpful assistant and have knowledge of Python. Your response should be in JSON format.'},
    {"role": "user", "content": prompt}
]
completion = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=messages,
    temperature=0.8,
    response_format={"type": "json_object"}
)

contents = []
contents.extend([choice.message.content for choice in completion.choices])

print(contents)

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

'''
'Plan 1: Dividing Labels Based on Natural Habitat
Category 1: Animals\n- Bird\n- Cat\n- Dog\n\nCategory 2: Man-Made Objects\n- Airplane
Plan 2: Dividing Labels Based on Movement Ability
Category 1: Animals\n- Bird\n- Cat\n- Dog\n\nCategory 2: Modes of Transportation\n- Airplane'
'''

# labels = ['airplane', 'bird', 'cat', 'dog']
# s = "" + f"'{labels[0]}'"
# for i in range(1, len(labels)):
#     s += f", '{labels[i]}'"

# a = torch.tensor([1., 1.])
# b = torch.tensor([2., 2.])
# c = (a + b) / 2
# d = torch.stack([a, b]).mean(dim=0)

# contents = ["Plan 1:\nSet 1(set name):1,2\nSet 2(set name):0,3\n\nPlan 2:\nSet 1(set name):0,1,2\nSet 2(set name):3\n\n"]
# plans = []
# for content in contents:
#     content = content.split('\n\n')[: -2]
#     for s in content:
#         categories = s.split('\n')[1:]
#         c = {}
#         for item in categories:
#             item = item.split('=')
#             name = item[0].split('(')[-1][:-2].strip()
#             c[name] = list(item[-1])
#             for l in item[1:]:
#                 l = l.strip()
#                 c[name].append(l)
#         plans.append(c)
# print(plans)

a = contents[0]
a = a.replace('\n', '')
a = json.loads(a)
print(a)
'''
{
    "plan1": {
        "set1": {
            "class": "animals",
            "words": ["2", "3"]
        },
        "set2": {
            "class": "vehicles",
            "words": ["0", "1"]
        }
    },
    "plan2": {
        "set1": {
            "class": "flying",
            "words": ["0", "1"]
        },
        "set2": {
            "class": "four-legged",
            "words": ["2", "3"]
        }
    }
}
'''
