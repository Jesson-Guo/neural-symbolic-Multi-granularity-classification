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
Input: {151: 'chihuahua.n.03', 152: 'japanese_spaniel.n.01', 153: 'maltese_dog.n.01', 154: 'pekinese.n.01', 155: 'shih-tzu.n.01', 157: 'papillon.n.01', 158: 'toy_terrier.n.01', 159: 'rhodesian_ridgeback.n.01', 161: 'basset.n.01', 162: 'beagle.n.01', 163: 'bloodhound.n.01', 164: 'bluetick.n.01', 165: 'black-and-tan_coonhound.n.01', 166: 'walker_hound.n.01', 167: 'english_foxhound.n.01', 168: 'redbone.n.01', 179: 'staffordshire_bullterrier.n.01', 180: 'american_staffordshire_terrier.n.01', 181: 'bedlington_terrier.n.01', 182: 'border_terrier.n.01', 183: 'kerry_blue_terrier.n.01', 184: 'irish_terrier.n.01', 185: 'norfolk_terrier.n.01', 186: 'norwich_terrier.n.01', 187: 'yorkshire_terrier.n.01', 188: 'wire-haired_fox_terrier.n.01', 189: 'lakeland_terrier.n.01', 190: 'sealyham_terrier.n.01', 191: 'airedale.n.01', 192: 'cairn.n.02', 193: 'australian_terrier.n.01', 194: 'dandie_dinmont.n.01', 195: 'boston_bull.n.01', 196: 'miniature_schnauzer.n.01', 197: 'giant_schnauzer.n.01', 198: 'standard_schnauzer.n.01', 199: 'scotch_terrier.n.01', 200: 'tibetan_terrier.n.01', 201: 'silky_terrier.n.01', 202: 'soft-coated_wheaten_terrier.n.01', 203: 'west_highland_white_terrier.n.01', 204: 'lhasa.n.02', 215: 'brittany_spaniel.n.01', 216: 'clumber.n.01', 217: 'english_springer.n.01', 218: 'welsh_springer_spaniel.n.01', 219: 'cocker_spaniel.n.01', 220: 'sussex_spaniel.n.01', 221: 'irish_water_spaniel.n.01', 252: 'affenpinscher.n.01', 253: 'basenji.n.01', 254: 'pug.n.01', 259: 'pomeranian.n.01', 260: 'chow.n.03', 261: 'keeshond.n.01', 262: 'brabancon_griffon.n.01', 263: 'pembroke.n.01', 264: 'cardigan.n.02', 265: 'toy_poodle.n.01', 266: 'miniature_poodle.n.01', 267: 'standard_poodle.n.01', 268: 'mexican_hairless.n.01'}\
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
