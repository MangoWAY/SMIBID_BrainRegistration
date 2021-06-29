import torch as th
import numpy as np


def compute_grid(image_size, dtype=th.float32, device='cpu'):

    dim = len(image_size)

    if dim == 2:
        nx = image_size[0]
        ny = image_size[1]

        x = th.linspace(-1, 1, steps=ny).to(dtype=dtype)
        y = th.linspace(-1, 1, steps=nx).to(dtype=dtype)

        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return th.cat((x, y), 3).to(dtype=dtype, device=device)

    elif dim == 3:
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]

        x = th.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = th.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = th.linspace(-1, 1, steps=nz).to(dtype=dtype)

        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)

        return th.cat((x, y, z), 4).to(dtype=dtype, device=device)


def warp_image(image, dvf, mode='bilinear'):
    grid = compute_grid(image.shape,device="cuda:0")
    image = th.from_numpy(image).unsqueeze(0).unsqueeze(0).to("cuda:0")
    return th.nn.functional.grid_sample(image, grid + dvf.unsqueeze(0), mode=mode).squeeze(0).squeeze(0)