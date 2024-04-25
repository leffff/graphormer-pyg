# graphormer-pyg
[Microsoft Graphormer](https://github.com/microsoft/Graphormer) rewritten in PyTorch-Geometric

# !!! Dear Developers !!!
I currently do not have time to modify or work on this repository.</br>
This implementation has one main disadvantage: IT's SLOW!</br>
During the past few months I have recieved numerous issues concerning the execution speed.</br>
If anyone wants to speedup my code or work on further development of this repository, you are MORE THAN WELCOME!</br>
I will review all of your PRs and we can discuss any issues you are willing to discuss!</br>


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
