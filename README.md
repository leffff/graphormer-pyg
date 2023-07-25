# graphormer-pyg
[Microsoft Graphormer](https://github.com/microsoft/Graphormer) rewritten in PyTorch-Geometric

![image](https://github.com/leffff/graphormer-pyg/assets/57654885/34c1626e-aa71-4f2a-a12c-0d5900d32cbf)

# Implemented Layers
1. Centrality Encoding
2. Spatial Encoding
3. Edge Encoding
4. Multi-Head Self-Attention

# Warning
This implementation differs from the original implementation in the paper in following ways:
1. No [VNode] ([CLS] token analogue in BERT)
2. The shortest path algorithm is not mentioned in the paper. This repository uses Floyd-Warshall
