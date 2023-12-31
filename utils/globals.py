import copy
from src.hierarchy import get_full_hierarchy, Node


_global_dict = {
    'label2id': {},
    'lpaths': {},
    'tree': None
}


def init():
    global _global_dict
    _global_dict = {
        'label2id': {},
        'lpaths': {},
        'tree': None
    }


def set_value(key, value):
    _global_dict[key] = value
 
 
def get_value(key):
    return _global_dict[key]


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


def init_global(args, class_to_idx):
    words = {}
    wnids = []
    label2id = {}
    id2label = {}
    lpaths = {}
    if args.arch == 'cifar10' or args.arch == 'cifar100':
        wnids = open(args.wnids, 'r')
        wnids = ''.join(wnids.readlines()).split()
        f = open(args.words, 'r')
        for line in f.readlines():
            line = line.split('\n')[0]
            line = line.split('\t')
            words[line[0]] = line[1]
        for wnid in wnids:
            label2id[wnid] = class_to_idx[words[wnid]] + 1
            id2label[class_to_idx[words[wnid]] + 1] = wnid
    elif args.arch == 'tiny-imagenet' or args.arch == 'imagenet':
        for k, v in class_to_idx.items():
            wnids.append(k)
            label2id[k] = v + 1
            id2label[v+1] = k

    _, dic = get_full_hierarchy(args.hier)
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
                leaf.parent = temp
                node = node.parent
                leaf = temp
            else:
                temp = node_dict[node.parent.wnid]
                temp.update_child(leaf)
                leaf.parent = temp
                break
    node_dict = prunning_tree(node_dict)

    # init label2id and lpaths
    index = args.num_classes + 1
    for wnid in wnids:
        node = node_dict[wnid]
        lpaths[label2id[wnid]] = [label2id[wnid]]
        while node.parent != None:
            node = node.parent
            if node.wnid not in label2id.keys():
                label2id[node.wnid] = index
                id2label[index] = node.wnid
                index += 1
            lpaths[label2id[wnid]].insert(0, label2id[node.wnid])

    lpaths_copy = copy.deepcopy(lpaths)
    for k, lp in lpaths.items():
        i = 1
        for v in lp:
            if v not in lpaths:
                lpaths_copy[v] = lpaths[k][:i]
            i += 1
    lpaths = lpaths_copy

    node_hash = {-1: 0}
    for wnid, node in node_dict.items():
        if not node.is_leaf():
            node_hash[label2id[wnid]] = len(node.children) + 1
    last_k = -1
    for k, x in node_hash.items():
        node_hash[k] = node_hash[last_k] + x
        last_k = k
    del node_hash[-1]

    return label2id, id2label, lpaths, node_dict['fall11'], node_dict, node_hash, node_hash[last_k]
