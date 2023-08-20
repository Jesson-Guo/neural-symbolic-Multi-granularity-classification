import xml.etree.cElementTree as ET
import cv2
import os
import numpy as np
import copy


class Node(object):
    def __init__(self, id, parent=None):
        self.id = id
        self.parent = parent
        self.children = {}
        self.nchild = 0 # num of children
        self.ncount= 0 # num of ImageNet1K children

    def update_child(self, node):
        self.children[self.nchild] = node
        self.nchild += 1

    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False


def pruning(node):
    """
    Pruning the total tree, only reserve the subtrees that contrains 1000 classes.
    """
    queue = [node]
    pid  = [0]
    while len(queue)>0:
        x = queue.pop(0)
        id = pid.pop(0)
        if x.ncount == 0:
            x.parent.children.pop(id)
        else:
            for k,v in x.children.items():
                queue.append(v)
                pid.append(k)


def pruning_count(node, dic):
    """
    Counting the the number of leaves that are contained in ImageNet1K.
    """
    if node.is_leaf():
        if not node.id in dic:
            node.ncount = 0
            return 0
        else:
            node.ncount = 1
            return 1
    else:
        count = 0
        for i in node.children.keys():
            count += pruning_count(node.children[i], dic)
        node.ncount = count
        return count


def get_full_hierarchy(file):
    tree = ET.ElementTree(file=file)
    root = tree.getroot()

    synset = root[1]
    dic = {'fall11':Node('fall11')}
    tree = dic['fall11']

    queue = [synset]

    while len(queue) > 0:
        node = queue.pop(0)
        cur_tree = dic[node.attrib['wnid']]
        for c in node:
            queue.append(c)
            wnid = c.attrib['wnid']
            dic[wnid] = Node(wnid, cur_tree)
            cur_tree.update_child(dic[wnid])

    return dic['fall11'], dic


def get_hierarchy(hier_path, wnids):
    tree, dic = get_full_hierarchy(hier_path)
    pruning_count(tree, wnids)
    pruning(tree)

    labels = {}
    for wnid in wnids:
        node = dic[wnid]
        labels[wnid] = []
        while node.parent != None:
            node = node.parent
            labels[wnid].append(node.id)

    return tree, labels


def image_info():
    info = {}
    folder = './tiny-imagenet-200/train/'
    for d in os.listdir(folder):
        if d[0] != '.':
            f = os.path.join(os.path.join(folder, d), d+'_boxes.txt')
            f = open(f, 'r')
            info[d] = {'mean': 0., 'std': 0., 'num': 0}
            for line in f.readlines():
                split_line = line.split('\t')
                img_path = os.path.join(os.path.join(folder, d), 'images/'+split_line[0])
                x = cv2.imread(img_path).astype(np.float32) / 255
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                info[d]['mean'] += x.mean()
                info[d]['std'] += x.std()
                info[d]['num'] += 1
    # for d in info.keys():
    #     info[d]['mean'] /= info[d]['num']
    #     info[d]['std'] /= info[d]['num']
    return info


if __name__ == "__main__":
    hier_path = './structure_released.xml'
    wnid_path = './wnid.txt'
    wnid = open(wnid_path, 'r')
    wnids = ''.join(wnid.readlines()).split()

    tree, dic = get_full_hierarchy(hier_path)
    pruning_count(tree, wnids)
    pruning(tree)
    labels = {}
    for wnid in wnids:
        node = dic[wnid]
        labels[wnid] = []
        while node.parent != None:
            node = node.parent
            labels[wnid].append(node.id)
    info = image_info()
    info_copy = copy.deepcopy(info)
    for k in info.keys():
        ancestors = labels[k]
        for anc in ancestors:
            if anc not in info_copy.keys():
                info_copy[anc] = {'mean': 0., 'std': 0., 'num': 0}
            info_copy[anc]['mean'] += info[k]['mean']
            info_copy[anc]['std'] += info[k]['std']
            info_copy[anc]['num'] += info[k]['num']
    for k in info.keys():
        info_copy[k]['mean'] /= info_copy[k]['num']
        info_copy[k]['std'] /= info_copy[k]['num']
    print()
