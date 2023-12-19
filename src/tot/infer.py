from src.tot.tot import Thought


def solve(model, dataloader, tree, node_dict, val_loader, gpt, metrics_func):
    inner_nodes = {}
    leaves = []
    for node in node_dict.values():
        if node.is_leaf():
            leaves.append(node)
            continue
        if node.layer not in inner_nodes:
            inner_nodes[node.layer] = []
        inner_nodes[node.layer].append(node)

    root = Thought(labels=leaves, feedback=None, parent=None)

    for i, (x, targets) in enumerate(dataloader):
        x = model.forward_features(x)
        x = model.forward_head(x, pre_logits=False)
