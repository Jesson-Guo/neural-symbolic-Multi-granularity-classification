import torch
import progressbar

from src.tot.tot import ToT
from src.gpt import GPT
from utils.conf import is_main_process
from utils.util import accuracy, reduce_mean


def solve(model, dataloader, node_dict, label_to_wnid, label_to_id, gpt: GPT, tot: ToT):
    inner_nodes = {}
    leaves = []
    for node in node_dict.values():
        if node.is_leaf():
            leaves.append(node)
            continue
        if node.layer not in inner_nodes:
            inner_nodes[node.layer] = []
        inner_nodes[node.layer].append(node)

    acc = 0
    if is_main_process():
        bar = progressbar.ProgressBar(0, len(dataloader))
    for idx, (x, targets) in enumerate(dataloader):
        x = model.forward_features(x)
        x = model.forward_head(x, pre_logits=False)

        pred = torch.zeros(x.shape[0])
        for i in range(x.shape[0]):
            output, _ = tot.solve(x, dataloader.dataset.labels, node_dict, label_to_wnid, gpt, method='bfs')
            pred[i] = label_to_id[output]

        acc += pred.eq(targets.data).sum()

        if is_main_process():
            bar.update(i+1)

    acc = reduce_mean(acc, average=False)
    acc = acc / len(dataloader.dataset)
    if is_main_process():
        print(f'eval acc: {acc.item()}')
