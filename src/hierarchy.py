import xml.etree.cElementTree as ET


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
        self.prob = None
        self.path_prob = None
        self.subid = {}

    def update_child(self, node):
        self.children[self.nchild] = node
        self.nchild += 1
    
    def set_layer(self, layer):
        self.layer = layer

    def set_weight(self, weight):
        self.weight = weight

    def set_classifier(self, classifier):
        self.classifier = classifier

    def set_subid(self):
        keys_list = list(self.children.keys())
        for i in range(len(keys_list)):
            self.subid[self.children[keys_list[i]].wnid] = i+1

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
    tree = ET.ElementTree(file=file)
    root = tree.getroot()

    synset = root[1]
    dic = {'fall11':Node('fall11', 'Thing')}
    tree = dic['fall11']

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


def get_hierarchy(dic, words_path):
    f = open(words_path, 'r')
    words = {}
    for line in f.readlines():
        line = line.split('\n')[0]
        line = line.split('\t')
        words[line[0]] = line[1]

    node_dict = {}
    leaf = None
    for k, label in words.items():
        leaf = Node(k, label)
        node_dict[k] = leaf
        node = dic[k]
        while node.parent != None:
            if not node.parent.wnid in node_dict.keys():
                temp = Node(node.parent.wnid, node.parent.name)
                node_dict[node.parent.wnid] = temp
                temp.update_child(leaf)
                node = node.parent
                leaf = temp
            else:
                temp = node_dict[node.parent.wnid]
                temp.update_child(leaf)
                break
    return node_dict['fall11'], node_dict


if __name__ == "__main__":
    hier_path = './structure_released.xml'
    words_path = './words.txt'

    _, dic = get_full_hierarchy(hier_path)
    tree = get_hierarchy(dic, words_path)
    # pruning_count(tree, wnids)
    # pruning(tree)

    print()
