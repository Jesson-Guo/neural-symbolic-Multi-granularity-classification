import os
from shutil import copy2, rmtree
from os import rmdir

target_folder = './tiny-imagenet-200/val/'

val_dict = {}
with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        file = target_folder + 'images/' + split_line[0]
        box = split_line[2] + '\t' + split_line[3] + '\t' + split_line[4] + '\t' + split_line[5]
        if not os.path.exists(target_folder + split_line[1]):
            os.mkdir(target_folder + split_line[1])
            os.mkdir(target_folder + split_line[1] + '/images')
        copy2(file, target_folder + split_line[1] + '/images/')
        # move(target_folder + split_line[1] + '/images/' + split_line[0], file)
        if split_line[1] not in val_dict.keys():
            val_dict[split_line[1]] = []
        val_dict[split_line[1]].append(split_line[0]+'\t'+box+'\n')

for k in val_dict.keys():
    dir = target_folder + os.sep + k
    f = open(dir+os.sep+k+'_boxes.txt', 'w')
    for box in val_dict[k]:
        f.write(box)

rmtree('./tiny-imagenet-200/val/images')
