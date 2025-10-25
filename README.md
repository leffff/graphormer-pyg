# graphormer-pyg
[Microsoft Graphormer](https://github.com/microsoft/Graphormer) rewritten in PyTorch-Geometric

# Modifications made
Compared to the original implementation of [graphormer-pyg](https://github.com/leffff/graphormer-pyg) (commit 231fcf0), this implementation has the following modifications to improve execution speed:
1. The in_degree and out_degree, as used in the centrality encoding layer, are pre-computed and stored in the graph data structure. 
2. The length of the shortest path between each pair of nodes is pre-computed as a tensor and passed to the spatial encoding module, in which the original for loops are replaced by tensor operations.
3. The path between each pair of edges and the length of the path are pre-computed as tensors and passed to the edge encoding module, in which the original for loops are replaced by tensor operations.
</br>

In my tests on MoleculeNet dataset with GPU, the training speed was improved by about 50 fold.  


![image](https://github.com/leffff/graphormer-pyg/assets/57654885/34c1626e-aa71-4f2a-a12c-0d5900d32cbf)


