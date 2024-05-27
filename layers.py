import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        self.mode = mode
        #create xy grids
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids, dim=0)
        grid = torch.unsqueeze(grid, dim=0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:] #grid shapes

        #resample to [-1, 1] to use F.grid_sample()
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]] # swap 2 dim to use F.grid_sample()
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)

def test():
    TPS = SpatialTransformer((3,3))
    src = torch.randn(1, 2, 3, 3)
    flow = torch.Tensor([
        [[0, 0.1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0.1, 0], [0, 0, 0], [0, 0, 0]]
    ])
    flow.unsqueeze_(0)
    print(src, '\n', TPS(src, flow))

if __name__ == '__main__':
    test()
