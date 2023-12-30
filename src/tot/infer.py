import torch
import progressbar

from src.tot.tot import ToT
from src.gpt import GPT
from utils.conf import is_main_process
from utils.util import accuracy, reduce_mean


def solve(model, dataloader, node_dict, label_to_wnid, label_to_id, device, tot: ToT, alpha):
    head_weight = model.state_dict()['head.weight'].T
    acc = 0
    if is_main_process():
        bar = progressbar.ProgressBar(0, len(dataloader))
    for idx, (x, targets) in enumerate(dataloader):
        torch.cuda.empty_cache()
        x = x.to(device)
        targets = targets.to(device)

        x = model.forward_features(x)
        x = model.forward_head(x, pre_logits=True)

        # output = x.data @ head_weight.data
        # a = output.max(dim=0)
        # b = output.min(dim=0)
        # output = output.softmax(dim=1)
        # pred = output.data.max(1)[1]

        pred = torch.zeros(x.shape[0]).to(device)
        for i in range(x.shape[0]):
            _, label = tot.solve(x[i].data, node_dict, label_to_wnid, alpha, method='dfs')
            pred[i] = label_to_id[label]

        acc += pred.eq(targets.data).sum()
        targets.detach()
        x.detach()

        if is_main_process():
            bar.update(idx+1)

    acc = reduce_mean(acc, average=False)
    acc = acc / len(dataloader.dataset)
    if is_main_process():
        print(f'eval acc: {acc.item()}')
