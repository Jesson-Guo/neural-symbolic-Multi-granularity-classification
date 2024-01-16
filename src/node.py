import xml.etree.cElementTree as ET
import copy
from nltk.corpus import wordnet as wn


class Node(object):
    def __init__(self, wnid, name, gloss, parent=None):
        self.wnid = wnid
        self.name = name
        self.gloss = gloss
        self.layer = 0
        self.parent = parent
        self.children = {}
        self.nchild = 0
        self.weight = None

    def update_child(self, node):
        self.children[self.nchild] = node
        self.nchild += 1

    def set_weight(self, weight):
        self.weight = weight

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
    dic = {'fall11': Node('fall11', 'Thing', None)}

    queue = [synset]

    while len(queue) > 0:
        node = queue.pop(0)
        cur_tree = dic[node.attrib['wnid']]
        for c in node:
            queue.append(c)
            wnid = c.attrib['wnid']
            words = c.attrib['words']
            gloss = c.attrib['gloss']
            dic[wnid] = Node(wnid, words, gloss, cur_tree)
            cur_tree.update_child(dic[wnid])

    return dic['fall11'], dic


def prunning_tree(node_dict):
    temp = copy.deepcopy(node_dict)
    for i, node in node_dict.items():
        if node.parent == None:
            continue
        if len(node.children) == 1:
            child = node.children[0]
            parent = node.parent
            for k, v in parent.children.items():
                if v.wnid == node.wnid:
                    break
            parent.children[k] = child
            child.parent = parent
            del temp[i]
    prunned_dict = {}
    prunned_dict['fall11'] = node_dict['fall11']
    del node_dict['fall11']
    for i, node in node_dict.items():
        if i in temp:
            prunned_dict[i] = node_dict[i]
    return prunned_dict


def init_weight(node: Node, layer):
    node.layer = layer
    if node.is_leaf():
        return node.weight
    weight = 0
    for v in node.children.values():
        weight += init_weight(v, layer+1)
    weight /= node.num_child()
    node.set_weight(weight)
    return weight


def build_tree(args, class_to_idx):
    class_to_idx_t = {}
    if args.data.startswith('cifar10') or args.data.startswith('cifar100'):
        f = open(args.words, 'r')
        for line in f.readlines():
            line = line.split('\n')[0]
            line = line.split('\t')
            class_to_idx_t[line[0]] = class_to_idx[line[1]]
        class_to_idx = class_to_idx_t
        del class_to_idx_t

    _, dic = get_full_hierarchy(args.hier)
    node_dict = {}
    leaf = None
    for wnid, idx in class_to_idx.items():
        node = dic[wnid]
        name = wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:])).name()
        leaf = Node(wnid, name, node.gloss)
        # leaf.set_weight(weights[idx])
        node_dict[wnid] = leaf
        while node.parent != None:
            if not node.parent.wnid in node_dict.keys():
                # name = wn.synset_from_pos_and_offset(node.parent.wnid[0], int(node.parent.wnid[1:])).name()
                temp = Node(node.parent.wnid, node.parent.name, node.parent.gloss)
                node_dict[node.parent.wnid] = temp
                temp.update_child(leaf)
                leaf.parent = temp
                node = node.parent
                leaf = temp
            else:
                temp = node_dict[node.parent.wnid]
                temp.update_child(leaf)
                leaf.parent = temp
                break
    node_dict = prunning_tree(node_dict)
    tree = node_dict['fall11']
    # init_weight(tree, 0)

    label_to_wnid, label_to_id = {}, {}
    labels = {}
    for k, v in node_dict.items():
        label_to_wnid[v.name] = k

    i = 0
    for k, v in class_to_idx.items():
        parent_name = node_dict[k].name
        if parent_name not in label_to_id:
            label_to_id[parent_name] = v
            labels[i] = parent_name
            i += 1

    node_children = {}
    for wnid, idx in class_to_idx.items():
        node = node_dict[wnid]
        while node.parent != None:
            if node.wnid not in node_children:
                node_children[node.wnid] = {}
            node_children[node.wnid][idx] = node_dict[wnid].name
            node = node.parent

    return node_dict, label_to_wnid, label_to_id, labels, node_children
