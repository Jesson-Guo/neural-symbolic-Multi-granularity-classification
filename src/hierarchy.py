import xml.etree.cElementTree as ET


lpaths = {}
label2id = {}


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


def get_hierarchy(args):
    _, dic = get_full_hierarchy(args.hier)

    wnids = open(args.wnids, 'r')
    wnids = ''.join(wnids.readlines()).split()

    # init label2id and lpaths
    index = 1
    for wnid in wnids:
        label2id[wnid] = index
        index += 1
    for wnid in wnids:
        node = dic[wnid]
        lpaths[label2id[wnid]] = [label2id[wnid]]
        while node.parent != None:
            node = node.parent
            if node.wnid not in label2id.keys():
                label2id[node.wnid] = index
                index += 1
            lpaths[label2id[wnid]].insert(0, label2id[node.wnid])

    # init tree
    node_dict = {}
    leaf = None
    for wnid in wnids:
        node = dic[wnid]
        leaf = Node(wnid, node.name)
        node_dict[wnid] = leaf
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
