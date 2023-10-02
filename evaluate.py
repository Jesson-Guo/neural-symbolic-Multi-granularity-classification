import torch
import progressbar
import copy

from src.utils import *


@torch.no_grad()
def evaluate(dataloader, model, inference, lpaths, conf, device):
    model.eval()
    inference.eval()
    accuracy = AverageMeter()
    bar = progressbar.ProgressBar(0, len(dataloader))

    # for images, labels, boxes in dataloader:
    for i, (x, targets) in enumerate(dataloader):
        x = torch.autograd.Variable(x)
        x = x.to(device)
        labels = targets['labels']
        labels = labels.to(device)

        feat = model(x)
        feat_list = [feat['0'], feat['1'], feat['2'], feat['3']]
        out = inference.infer(feat_list, device=device)

        for k in range(len(out)):
            infer_path, diffs = out[k]
            gt_path = copy.deepcopy(lpaths[labels[k].item()])
            gt_path.reverse()
            for j in range(min(len(gt_path), len(infer_path))):
                if infer_path[j] != gt_path[j] or diffs[j] < conf:
                    break
            accuracy.update(j/(len(gt_path)-1.0))
            del gt_path

        bar.update(i)

    return accuracy.avg
