import torch
import progressbar

from src.tot.tot import ToT
from src.gpt import GPT
from utils.conf import is_main_process
from utils.util import accuracy, reduce_mean


def solve(model, dataloader, node_dict, label_to_wnid, label_to_id, labels, device, gpt: GPT, tot: ToT):
    acc = 0
    if is_main_process():
        bar = progressbar.ProgressBar(0, len(dataloader))
    for idx, (x, targets) in enumerate(dataloader):
        x = x.to(device)
        targets = targets.to(device)

        x = model.forward_features(x)
        x = model.forward_head(x, pre_logits=True)

        pred = torch.zeros(x.shape[0]).to(device)
        for i in range(x.shape[0]):
            output, _, _ = tot.solve(x, labels, node_dict, label_to_wnid, gpt, method='bfs')
            pred[i] = label_to_id[output]

        acc += pred.eq(targets.data).sum()

        if is_main_process():
            bar.update(idx+1)

    acc = reduce_mean(acc, average=False)
    acc = acc / len(dataloader.dataset)
    if is_main_process():
        print(f'eval acc: {acc.item()}')
