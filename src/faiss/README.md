# Faiss

Faiss的使用说明和相关知识。

## Composite Indexes

一个复合索引可由以下部分组合而成：

- **Vector transform**：在索引之前应用于向量的预处理步骤
  - **PCA**：主成分分析，尽可能减少信息损失，将向量投影到低维空间
  - **OPQ**：通过旋转变换优化向量分布，避免信息分布不均匀，以提高量化效果
- **Coarse quantizer**：用于限制搜索范围
  - **IVF**：
  - **IMI**：
  - **HNSW**：
- **Fine quantizer**：用于压缩索引大小
  - **PQ**：
- **Refinement**： 搜索结果排序方式
  - **RFlat**：在原始向量上计算距离
  
| **Vector transform** | **Coarse quantizer** | **Fine quantizer** | **Refinement** |
|----------------------|----------------------|--------------------|---------------|
| PCA, OPQ, RR, L2norm, ITQ, Pad | IVF, Flat, IMI, IVF-HNSW, IVF-PQ, IVF-RCQ, HNSW-Flat, HNSW-SQ, HNSW-PQ | Flat, **PQ**, **SQ**, *Residual*, RQ, LSQ, ZnLattice, LSH | RFlat, Refine* |

## Hierarchy

The CPU Faiss class hierarchy looks like this:[faiss_class_hierarchy.pdf](https://github.com/facebookresearch/faiss/wiki/faiss_class_hierarchy.pdf)

![Faiss Class Hierarchy](./img/faiss_class_hierarchy.svg)

> [Faiss: The Missing Manual](https://www.pinecone.io/learn/series/faiss/)