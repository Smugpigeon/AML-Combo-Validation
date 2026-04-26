"""v0.6 graph-based encoders.

drug_gin.py:
    GIN drug encoder — replaces nn.Embedding(165, 64) with a chemistry-grounded
    encoder over atom-bond molecular graphs. Built on PyTorch Geometric.

gene_ppi_gat.py (Phase 2):
    GAT over the STRING PPI subgraph for context-aware mutation embedding.
"""
