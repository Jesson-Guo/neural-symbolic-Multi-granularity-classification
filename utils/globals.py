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


def init_global(args, class_to_idx):
    _, dic = get_full_hierarchy(args.hier)

    words = {}
    wnids = []
    label2id = {}
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
    elif args.arch == 'tiny-imagenet' or args.arch == 'imagenet':
        for k, v in class_to_idx.items():
            wnids.append(k)
            label2id[k] = v + 1

    # init label2id and lpaths
    index = args.num_classes + 1
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
                leaf.parent = temp
                node = node.parent
                leaf = temp
            else:
                temp = node_dict[node.parent.wnid]
                temp.update_child(leaf)
                leaf.parent = temp
                break
    return label2id, lpaths, node_dict['fall11'], node_dict
