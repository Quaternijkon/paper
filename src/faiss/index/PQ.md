# Product Quantization

PQ并不是一种索引，而是创建复合索引中的一种Fine quantizer。

## Introduction

<iframe width="560" height="315" src="https://www.youtube.com/embed/t9mRf2S5vDI?si=-y4TyDmvil9q9cd5" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

向量相似性搜索可能需要大量内存，随着数据集大小的不断增加，内存占用会变得难以承受。PQ可以大幅压缩高维向量，甚至能减少97%的内存占用。（But at what cost？）

### Quantization & Dimensionality-reduction

量化与降维不同，降维的目的是得到一个低维向量，而量化的目的是得到一个低精度向量。

> [!TIP]
> 比如我有一个128维的FP32向量，我可以将其降维到64维（不是简单的裁切，比如使用PCA主成分分析），也可以将其量化为INT8向量。毫无疑问这两种方法都减少了向量的占用，而且都损失了信息。
> ![pqex](./img/pqex.jpg)
> 降维缩小了向量的维度，量化缩小了能表示的范围。
> 但是PQ并不是简单的将每个维度的信息量化成更低精度的表示。粗略地说，PQ先将向量分成几个子向量，每个子向量用它们所在的簇的ID来表示。

### How PQ works

1. 我有一些128维的向量。
2. 按32维切分成4个子向量。
3. 对所有128维向量的相同位置上的32维子向量进行聚类，例如得到128个簇。
4. 用每个子向量所在簇的ID来表示这个子向量。（这样我们就能用2字节表示一个子向量）
5. 最终128维的向量可以用8字节表示。

![pq](./img/pq.png)

1. 切分向量
```python
x = [1, 8, 3, 9, 1, 2, 9, 4, 5, 4, 6, 2]

m = 4
D = len(x)
# ensure D is divisable by m
assert D % m == 0
# length of each subvector will be D / m (D* in notation)
D_ = int(D / m)

# now create the subvectors
u = [x[row:row+D_] for row in range(0, D, D_)]
print(u)
```
2. 创建聚类
```python
k = 2**5
assert k % m == 0
k_ = int(k/m)
print(f"{k=}, {k_=}")

from random import randint

c = []  # our overall list of reproduction values
for j in range(m):
    # each j represents a subvector (and therefore subquantizer) position
    c_j = []
    for i in range(k_):
        # each i represents a cluster/reproduction value position *inside* each subspace j
        c_ji = [randint(0, 9) for _ in range(D_)]
        c_j.append(c_ji)  # add cluster centroid to subspace list
    # add subspace list of centroids to overall list
    c.append(c_j)
```

```python
def euclidean(v, u):
    distance = sum((x - y) ** 2 for x, y in zip(v, u)) ** .5
    return distance

def nearest(c_j, u_j):
    distance = 9e9
    for i in range(k_):
        new_dist = euclidean(c_j[i], u_j)
        if new_dist < distance:
            nearest_idx = i
            distance = new_dist
    return nearest_idx

ids = []
for j in range(m):
    i = nearest(c[j], u[j])
    ids.append(i)
print(ids)
```

```python
q = []
for j in range(m):
    c_ji = c[j][ids[j]]
    q.extend(c_ji)

print(q)
```

~~~admonish success collapsible=true, title='完整代码'
```python
x = [1, 8, 3, 9, 1, 2, 9, 4, 5, 4, 6, 2]

m = 4
D = len(x)
# ensure D is divisable by m
assert D % m == 0
# length of each subvector will be D / m (D* in notation)
D_ = int(D / m)

# now create the subvectors
u = [x[row:row+D_] for row in range(0, D, D_)]
print(u)

k = 2**5
assert k % m == 0
k_ = int(k/m)
print(f"{k=}, {k_=}")

from random import randint

c = []  # our overall list of reproduction values
for j in range(m):
    # each j represents a subvector (and therefore subquantizer) position
    c_j = []
    for i in range(k_):
        # each i represents a cluster/reproduction value position *inside* each subspace j
        c_ji = [randint(0, 9) for _ in range(D_)]
        c_j.append(c_ji)  # add cluster centroid to subspace list
    # add subspace list of centroids to overall list
    c.append(c_j)
    
def euclidean(v, u):
    distance = sum((x - y) ** 2 for x, y in zip(v, u)) ** .5
    return distance

def nearest(c_j, u_j):
    distance = 9e9
    for i in range(k_):
        new_dist = euclidean(c_j[i], u_j)
        if new_dist < distance:
            nearest_idx = i
            distance = new_dist
    return nearest_idx

ids = []
for j in range(m):
    i = nearest(c[j], u[j])
    ids.append(i)
print(ids)

q = []
for j in range(m):
    c_ji = c[j][ids[j]]
    q.extend(c_ji)

print(q)
```
~~~