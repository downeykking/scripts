import numpy as np


def alias_setup(probs):
    """建立alias table
    Args:
        (list): 要求的概率分布数组, 需要是sort为descending后再进入采样
    Returns:
        (tuple of list): 返回alias数组与prob数组
    """
    n = len(probs)
    # prob数组
    prob = np.zeros(n)
    # alias数组
    alias = np.zeros(n, dtype=np.int64)
    # that are larger and smaller than 1/n.
    # 存储比1小的列
    smaller = []
    # 存储比1大的列
    larger = []
    for i, p in enumerate(probs):
        prob[i] = n * p
        if prob[i] < 1.0:
            smaller.append(i)
        else:
            larger.append(i)

    # 通过拼凑，将各个类别都凑为1
    while len(smaller) > 0 and len(larger) > 0:
        small_idx = smaller.pop()
        large_idx = larger.pop()

        # 填充Alias数组
        alias[small_idx] = large_idx
        # 将大的分到小的上
        prob[large_idx] = prob[large_idx] - (1.0 - prob[small_idx])

        if prob[large_idx] < 1.0:
            smaller.append(large_idx)
        else:
            larger.append(large_idx)

    while larger:
        large_idx = larger.pop()
        prob[large_idx] = 1
    while smaller:
        small_idx = smaller.pop()
        prob[small_idx] = 1

    return alias, prob


def alias_sample(alias, prob):
    """进行采样
    Args:
        alias(list): alias数组
        prob(list): prob数组
    Returns:
        (int): 返回采样结果
    """
    n = len(alias)
    # 1. 取[0, n-1]的某一列
    i = int(np.random.rand() * n)
    # 2. 取[0, 1]的某个数
    r = np.random.rand()
    if r < prob[i]:
        return i
    else:
        return alias[i]


def simulate(N=30, k=10000):
    """模拟采样

    Args:
        N (int, optional): 需要的概率分布的总数. Defaults to 30.
        k (int, optional): 采样次数. Defaults to 10000.

    Returns:
        (tuple of list): 模拟概率分布, 真实概率分布
    """
    # Get a random probability vector.
    probs = np.random.dirichlet(np.ones(N), 1).ravel()

    # Construct the table.
    alias, prob = alias_setup(probs)

    # Generate variates.
    ans = np.zeros(N)
    for _ in range(k):
        i = alias_sample(alias, prob)
        ans[i] += 1

    return ans / np.sum(ans), probs


ans = simulate()
print(ans)
