from src.tot.tot import ToT
from src.gpt import GPT


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

    for idx, (x, targets) in enumerate(dataloader):
        x = model.forward_features(x)
        x = model.forward_head(x, pre_logits=False)

        for i in range(x.shape[0]):
            output, _ = tot.solve(x, dataloader.dataset.labels, node_dict, label_to_wnid, gpt, method='bfs')
            output = label_to_id[output]
