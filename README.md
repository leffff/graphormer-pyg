# graphormer-pyg
[Microsoft Graphormer](https://github.com/microsoft/Graphormer) rewritten in PyTorch-Geometric

# Warning
This implementation differs from the original implementation in the paper in following ways:
1. No [VNode] ([CLS] token analogue in BERT)
2. The shortest path algorithm is not mentioned in the paper. This repository uses DFS
