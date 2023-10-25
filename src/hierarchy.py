import xml.etree.cElementTree as ET
import torch
from utils.globals import *


class Node(object):
    def __init__(self, wnid, name, parent=None):
        self.wnid = wnid
        self.name = name
        self.layer = 0
        self.parent = parent
        self.children = {}
        self.nchild = 0 # num of children
        self.ncount= 0 # num of ImageNet1K children
        self.weight = None
        self.classifier = None
        self.sub_out = None
        self.prob = None
        self.path_prob = None
        self.feature = None
        self.subid = {}

    def node_distance(self, node):
        lp1 = get_value('lpaths')[self.wnid]
        lp2 = get_value('lpaths')[node.wnid]

        i, j = 0, 0
        while i < len(lp1) and j < len(lp2) and lp1[i] == lp2[j]:
            i += 1
            j += 1

        return (len(lp1)-1-i) + (len(lp2)-1-j)

    def update_child(self, node):
        self.children[self.nchild] = node
        self.nchild += 1
    
    def set_layer(self, layer):
        self.layer = layer

    def set_weight(self, weight):
        self.weight = weight

    def set_classifier(self, classifier):
        self.classifier = classifier

    def set_subid(self, label2id):
        for i, child in self.children.items():
            self.subid[label2id[child.wnid]] = i+1

    def get_subid(self, l):
        if l not in self.subid.keys():
            return 0
        else:
            return self.subid[l]

    def num_child(self):
        return len(self.children.keys())

    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False


def get_full_hierarchy(file):
    root = ET.ElementTree(file=file)
    root = root.getroot()

    synset = root[1]
    dic = {'fall11':Node('fall11', 'Thing')}

    queue = [synset]

    while len(queue) > 0:
        node = queue.pop(0)
        cur_tree = dic[node.attrib['wnid']]
        for c in node:
            queue.append(c)
            wnid = c.attrib['wnid']
            words = c.attrib['words']
            dic[wnid] = Node(wnid, words, cur_tree)
            cur_tree.update_child(dic[wnid])

    return dic['fall11'], dic
