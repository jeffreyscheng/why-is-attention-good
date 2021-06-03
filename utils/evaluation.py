from typing import List
from itertools import combinations
import numpy as np
import torch
import time


def gini(x) -> float:
    # tick = time.time()
    assert len(x.size()) == 1
    x = torch.nn.functional.softmax(x, dim=0).tolist()
    x = np.array(x, dtype=np.float32)
    n = len(x)
    diffs = sum(abs(i - j) for i, j in combinations(x, r=2))
    # print(time.time() - tick)
    return diffs / (2 * n**2 * x.mean())


def batched_gini(x):
    ginis = [gini(ex) for ex in torch.unbind(x, dim=0)]
    return sum(ginis) / len(ginis)
