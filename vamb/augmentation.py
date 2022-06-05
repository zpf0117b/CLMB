import numpy as np
import torch as torch
import random
import copy

__all__ = ['WordDrop','NumericalChange','GaussianNoise','FragmentTransfer','Reverse']

class GaussianNoise():
    def __init__(self):
        pass
        
    def __call__(self, sequence, mode):
        def get_parameter(m, mode):
            para = [[m*0.005 if m != 0 else 0.005, m/100 if m != 0 else 1/100, m/100 if m != 0 else 1/100, m/100 if m != 0 else 1/100, m/100 if m != 0 else 1/100, m/100 if m != 0 else 1/100],
                    [m/100 if m != 0 else 1/100, None, None, None, None, None],
                    [m/100 if m != 0 else 1/100, None, None, None, None, None],
                    [m/100 if m != 0 else 1/100, None, None, None, None, None],
                    [m/100 if m != 0 else 1/100, None, None, None, None, None],
                    [m/100 if m != 0 else 1/100, None, None, None, None, None]] # scale
            #return para[mode[0]][mode[1]]
            return para[mode[0]][mode[1]]
        var, m = torch.var(sequence), torch.mean(sequence)
        gen = torch.Generator(device=var.device)
        gen.manual_seed(42)
        #gen.seed()
        gaussian = torch.normal(torch.tensor(0), torch.sqrt(var), size=sequence.shape, generator=gen)
        noise = torch.divide(gaussian, torch.max(gaussian) / get_parameter(m, mode))
        #print("noise:", torch.sum(torch.pow(noise,2)))
        return torch.add(sequence, noise)
        # aug_sequence = torch.full_like(sequence, 0)
        # for i in range(sequence.shape[0]):
        #     torch.seed()
        #     var, m = torch.var_mean(sequence[i])
        #     gaussian = (torch.normal(0., float(torch.sqrt(var)), size=(1,sequence.shape[1])))[0]
        #     noise = torch.divide(gaussian, torch.max(gaussian)/get_parameter(m, mode))
        #     aug_sequence[i] = torch.add(sequence[i], noise)
        #     # if i % 10000 == 0:
        #     #     print(m, torch.max(noise))
        # # print(torch.sum(torch.pow(torch.sub(aug_sequence,sequence), 2)))
        # return aug_sequence

class NumericalChange():
    def __init__(self):
        pass
        
    def __call__(self, sequence, mode):
        para = [[None, 0.008, None, None, None, None],
                [0.008, 0.008, 0.008, 0.008, 0.008, 0.008],
                [None, 0.008, None, None, None, None],
                [None, 0.008, None, None, None, None],
                [None, 0.008, None, None, None, None],
                [None, 0.008, None, None, None, None]] # alpha
        aug_sequence = torch.full_like(sequence, 0)
        for i in range(sequence.shape[0]):
            torch.seed()
            a1, a2 = torch.bernoulli(torch.full_like(sequence[i], 0.7)), torch.bernoulli(torch.full_like(sequence[i], 0.3))
            a2 = torch.add(-a2,1.0)
            b = torch.mul(torch.sub(torch.add(a1,a2), 1),para[mode[0]][mode[1]])
            aug_sequence[i] = torch.mul(sequence[i], torch.add(b,1.0))
        return aug_sequence


class WordDrop():
    def __init__(self):
        pass
        
    def __call__(self, sequence, mode):
        para = [[None, None, 0.01, None, None, None],
                [None, None, 0.01, None, None, None],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [None, None, 0.01, None, None, None],
                [None, None, 0.01, None, None, None],
                [None, None, 0.01, None, None, None]] # cut
        aug_sequence = torch.full_like(sequence, 0)
        for i in range(sequence.shape[0]):
            gen = torch.Generator(device=sequence.device)
            gen.manual_seed(42)
            b = torch.bernoulli(torch.full_like(sequence[i], 1-para[mode[0]][mode[1]]), generator=gen)
            aug_sequence[i] = torch.mul(sequence[i], b)
            # print(torch.sum(aug_sequence[i]-sequence[i]))
        return aug_sequence

class FragmentTransfer():
    def __init__(self):
        pass
        
    def __call__(self, sequence, mode):
        # [left, right]
        para = [[None, None, None, 2, None, None],
                [None, None, None, 2, None, None],
                [None, None, None, 2, None, None],
                [2, 2, 2, 2, 2, 2],
                [None, None, None, 2, None, None],
                [None, None, None, 2, None, None]] # fragment
        dim = int(sequence.shape[1])
        span = list(torch.chunk(sequence[0], chunks=dim // para[mode[0]][mode[1]] + 1, dim=0))
        l = list(range(len(span)))
        random.shuffle(l)
        aug_sequence = torch.full_like(sequence, 0)
        for i in range(sequence.shape[0]):
            torch.seed()
            left, right = random.sample(l, 2)
            # for seq in sequence: seq=xxx cannot change the sequence
            span = list(torch.chunk(sequence[i], chunks=dim // para[mode[0]][mode[1]] + 1, dim=0))
            # shuffle_span = [i.numpy() for i in span]
            # random.shuffle(shuffle_span)
            # seq = torch.from_numpy(np.concatenate(shuffle_span,axis=0))
            # print(left,right)
            right = random.randint(left - para[mode[0]][mode[1]], left + para[mode[0]][mode[1]])
            span.insert(right if left > right else max(right-1, 0), span.pop(left))
            # print(span)
            aug_sequence[i] = torch.cat(span, dim=0)
        return aug_sequence

class Reverse():
    def __init__(self):
        pass
        
    def __call__(self, sequence, mode, windowsize=5):
        # [pos-left, pos+right]
        para = [[None, None, None, None, 1, None],
                [None, None, None, None, 1, None],
                [None, None, None, None, 1, None],
                [None, None, None, None, 1, None],
                [1, 1, 1, 1, 1, 1],
                [None, None, None, None, 1, None]] # windowsize
        dim = int(sequence.shape[1])
        # l = list(range(dim))
        # random.shuffle(l)
        # for k in range(sequence.shape[0]):
        #     random.seed(k)
        #     # left, right = random.sample(l, 2)
        #     if left < right:
        #         b = torch.cat((sequence[k][:left], torch.flip(sequence[k][left:right], dims=[0]), sequence[k][right:]), dim=0)
        #         # print('b1 ',b - sequence[k],left,right)
        #         sequence[k] = b
        #     else:
        #         b = torch.cat((torch.flip(sequence[k][:right], dims=[0]), sequence[k][right:left], torch.flip(sequence[k][left:], dims=[0])), dim=0)
        #         # print('b2 ',b - sequence[k],left,right)
        #         sequence[k] = b
        aug_sequence = torch.full_like(sequence, 0)
        for k in range(sequence.shape[0]):
            torch.seed()
            left = random.randint(0, para[mode[0]][mode[1]])
            right = para[mode[0]][mode[1]] - left
            pos = random.randint(left,dim-right-1)
            # print(pos,right,left)
            aug_sequence[k] = torch.cat((sequence[k][:pos-left], torch.flip(sequence[k][pos-left:pos], dims=[0]), torch.flip(sequence[k][pos:pos+right+1], dims=[0]), sequence[k][pos+right+1:]), dim=0)
        return aug_sequence

# class CropAndResize():
#     def __init__(self):
#         pass
        
#     def __call__(self, sequence, mode):
#         # [left, right]
#         dim = int(sequence.shape[1])
#         para = [[None, None, None, None, None, dim // 50 + 1],
#                 [None, None, None, None, None, dim // 50 + 1],
#                 [None, None, None, None, None, dim // 50 + 1],
#                 [None, None, None, None, None, dim // 50 + 1],
#                 [None, None, None, None, None, dim // 50 + 1],
#                 [dim // 50 + 1, dim // 50 + 1, dim // 50 + 1, dim // 50 + 1, dim // 50 + 1, 1]] # croped dimensions
#         # if crop_dim % 2 == 0:
#         #     left = crop_dim // 2
#         #     right = dim - 1 - crop_dim // 2
#         # else:
#         #     left = crop_dim // 2
#         #     right = dim - 1 - (crop_dim // 2 + 1)
#         # center = (left + right) / 2
#         # around_seq = [center + float((i - center) * dim / (dim - crop_dim)) 
#         #                             for i in range(dim)]
#         # crop_seq = [center + float((i - center) * dim / (dim - crop_dim)) 
#         #                             for i in range(dim)]
#         # for i in range(dim):
#         #     min_pos, min_dis = -1, dim
#         #     for j in range(left, right+1):
#         #         if abs(i - crop_seq[j]) < min_dis:
#         #             min_pos, min_dis = j, abs(i - crop_seq[j])
#         #     around_seq[i] = min_pos
        # # for k in range(sequence.shape[0]):
        # #     # print(sequence[k].dim(),sequence[k].shape)
        # #     a = torch.gather(sequence[k],0,torch.tensor(around_seq))
        # #     sequence[k] = a
        # # return sequence
        # # return torch.gather(sequence,dim=0,index=torch.tensor(around_seq))
        # around_seq_dict = {}
        # for left in range(0,para[mode[0]][mode[1]]+1):
        #     right = dim - 1 - (para[mode[0]][mode[1]] - left)
        #     center = (left + right) / 2
        #     around_seq_strech = [left + float((i - left) * (dim - left) / (dim - para[mode[0]][mode[1]])) 
        #                                 for i in range(dim)]
        #     # print(left,':',around_seq_strech[left:right+1],around_seq_strech)
        #     around_seq = [dim + float((i - dim) * dim / (dim - left)) 
        #                                 for i in around_seq_strech]
        #     crop_seq = copy.deepcopy(around_seq)           
        #     for i in range(dim):
        #         min_pos, min_dis = -1, 100000
        #         for j in range(left, right+1):
        #             if abs(i - crop_seq[j]) < min_dis:
        #                 min_pos, min_dis = j, abs(i - crop_seq[j])
        #         around_seq[i] = min_pos
        #     around_seq_dict[left] = around_seq
        #     # print(left,':',around_seq)
        #     # print(left,':',crop_seq[left:right+1],crop_seq)
            
        # aug_sequence = torch.full_like(sequence, 0)
        # for k in range(sequence.shape[0]):
        #     torch.seed()
        #     left = random.randint(0, para[mode[0]][mode[1]])
        #     a = torch.gather(sequence[k],0,torch.tensor(around_seq_dict[left]))
        #     aug_sequence[k] = a
        # # print(around_seq_dict)
        # return aug_sequence

# class GECA():
#     def __init__(self):
#         pass
        
#     def __call__(self, sequence, crop_dim=7):
        # [left, right]
        # dim = int(sequence.shape[1])
        # if crop_dim % 2 == 0:
        #     left = crop_dim // 2
        #     right = dim - 1 - crop_dim // 2
        # else:
        #     left = crop_dim // 2
        #     right = dim - 1 - (crop_dim // 2 + 1)
        # center = (left + right) / 2
        # around_seq = [center + float((i - center) * dim / (dim - crop_dim)) 
        #                             for i in range(dim)]
        # crop_seq = [center + float((i - center) * dim / (dim - crop_dim)) 
        #                             for i in range(dim)]
        # for i in range(dim):
        #     min_pos, min_dis = -1, dim
        #     for j in range(left, right+1):
        #         if abs(i - crop_seq[j]) < min_dis:
        #             min_pos, min_dis = j, abs(i - crop_seq[j])
        #     around_seq[i] = min_pos
        # for i in range(sequence.shape[0]):
        #     a = torch.full_like(sequence[i], i)
        #     for j in range(dim):
        #         a[j] = sequence[i][around_seq[j]]
        #     sequence[i] = a
        # return sequence
        # for a sequence, find a most similar, and replace others
        # use torch.where
def cli_main():
    a = FragmentTransfer()
    b = torch.randn((4,20),dtype=torch.float32)
    print(b)
    c = a(b,mode=(3,3))
    # print(b)
    print(c)
    print(c-b)

if __name__ == '__main__':
    cli_main()
