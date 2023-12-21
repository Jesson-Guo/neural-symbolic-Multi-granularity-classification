import torch.nn.functional as F
import torch.nn as nn
import torch
import openai
import os
import numpy as np
from torchmetrics.clustering import DunnIndex

from src.gpt import GPT


prompt = '''
Giving 2 plans to divide the INPUT into 2 categories with title in one word or one phrase.
INPUT: 'airplane', 'bird', 'cat', 'dog'
'''
# client = openai.OpenAI()
# res = chatgpt(client, 'say hello', model='gpt-3.5-turbo', temperature=0.7, max_tokens=1000, n=1, stop=None)
# print(res)
# completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "user", "content": prompt}
#     ]
# )

print()

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

labels = ['airplane', 'bird', 'cat', 'dog']
s = "" + f"'{labels[0]}'"
for i in range(1, len(labels)):
    s += f", '{labels[i]}'"

a = torch.tensor([1., 1.])
b = torch.tensor([2., 2.])
c = (a + b) / 2
d = torch.stack([a, b]).mean(dim=0)

plans = []
contents = ['Plan 1:\nCategory 1: Animals\n- Bird\n- Cat\n- Dog\n\nCategory 2: Transportation\n- Airplane\n\nPlan 2:\nCategory 1: Animals\n- Bird\n- Cat\n- Dog\n\nCategory 2: Flying Objects\n- Airplane']
for content in contents:
    content = content.replace('\n', ' ').strip()
    content = content.split('Plan')[1:]
    for plan in content:
        categories = plan.split('Category')[1:]
        c = {}
        for item in categories:
            item = item.split('-')
            name = item[0].split(':')[-1].strip()
            c[name] = []
            for l in item[1:]:
                c[name].append(l.strip())
        plans.append(c)
print(plans)
