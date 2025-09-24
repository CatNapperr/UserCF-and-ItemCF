结合M折交叉验证，简单实现基于用户和物品的协同过滤推荐算法

## 使用指南

### 1. 准备数据
下载 [MovieLens 1M](https://grouplens.org/datasets/movielens/) 数据集，  
解压后放置到代码所在根目录即可。

### 2. 运行 User-based CF
```bash
python usercf.py
```

### 3. 参数说明
在 usercf.py 与 itemcf.py 中可根据需要调整：

n_sim_user / n_sim_movie：相似邻居数量 K

n_rec_movie：推荐电影数量 N

pivot：训练/测试划分比例（仅 ItemCF 使用）
