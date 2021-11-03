import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch

def k_competition(topk, alpha, compete_tensor):
        # note: this operation only support those tensors that 
        # the differences between each values of them all are larger than 1e-15
        topk_pos, topk_pos_indices = torch.topk(compete_tensor, k=math.ceil(topk / 2), largest=True)
        topk_neg, topk_neg_indices = torch.topk(compete_tensor, k=int(topk / 2), largest=False)
        topk_neg[topk_neg > 0] = 0
        topk_pos[topk_pos < 0] = 0
        zero_p = torch.zeros_like(compete_tensor)

        pos = compete_tensor.clone().detach()
        pos[pos < 0] = 0
        # print(pos)
        bd_winner = torch.tensor(topk_pos.shape[1]-1)
        # if self.usecuda:
        #     bd_winner = bd_winner.cuda()
        pos_sep = torch.sub(pos, torch.sub(torch.index_select(topk_pos, 1, bd_winner), 1e-15))
        # avoid losing the boundary
        # print(pos_sep) 
        pos_lose = torch.where(pos_sep < 0,pos,zero_p)
        E_pos = torch.sum(pos_lose,dim=1,keepdim=True)
        pos = torch.add(pos,E_pos,alpha=alpha)
        pos_win = torch.where(pos_sep > 0,pos,zero_p)
        # print(pos_win)

        neg = compete_tensor.clone().detach()
        neg[neg > 0] = 0
        bd_winner = torch.tensor(topk_neg.shape[1]-1)
        # if self.usecuda:
        #     bd_winner = bd_winner.cuda()
        neg_sep = torch.sub(neg, torch.add(torch.index_select(topk_neg, 1, bd_winner),1e-15))
        # print(neg_sep)
        neg_lose = torch.where(neg_sep > 0,neg,zero_p)
        E_neg = torch.sum(neg_lose,dim=1,keepdim=True)
        neg = torch.add(neg,E_neg,alpha=alpha)
        neg_win = torch.where(neg_sep < 0,neg,zero_p)
        # print(neg_win)
        return pos_win + neg_win

X = np.array([[-1.0, -2.0, -3.0, 0.0, 2.0, 3., 4.], 
        [-3.0, -2.0, 1.0, 2.0, 3.0, 4., 5.], 
        [-1.0, 2.0, -3.0, 0.0, 1.0, 3., 4.],
        [-1.0, 0.0, -0.0, 0.0, 1.0, 3., 4.],
        [-1.0, -2.0, -3.0, 0.0, 0.0, 0., 4.],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6., 7.]])
Y = torch.tensor(X)
print(k_competition(5,0.1,Y)-Y)
