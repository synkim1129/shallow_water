import torch
import torch.nn.functional as F
import get_param

params = get_param.params()

def toCuda(x):
    if type(x) is tuple:
        return [xi.cuda() if params.cuda else xi for xi in x]
    return x.cuda() if params.cuda else x

def toCpu(x):
    if type(x) is tuple:
        return [xi.detach().cpu() for xi in x]
    return x.detach().cpu()

# First order derivatives (d/dx)
dx_kernel = toCuda(torch.Tensor([-0.5, 0, 0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
def dx(v, boundary_type='periodic'):
    """중앙차분법으로 x방향 미분"""
    if boundary_type == 'periodic':
        v_padded = F.pad(v, (1, 1, 0, 0), mode='circular')
    else:
        v_padded = F.pad(v, (1, 1, 0, 0), mode='replicate')
    return F.conv2d(v_padded, dx_kernel, padding=0)

# First order derivatives (d/dy)
dy_kernel = toCuda(torch.Tensor([-0.5, 0, 0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
def dy(v, boundary_type='periodic'):
    """중앙차분법으로 y방향 미분"""
    if boundary_type == 'periodic':
        v_padded = F.pad(v, (0, 0, 1, 1), mode='circular')
    else:
        v_padded = F.pad(v, (0, 0, 1, 1), mode='replicate')
    return F.conv2d(v_padded, dy_kernel, padding=0)

# Second order derivatives
# 1. x방향 2차 미분 (∂²/∂x²)
dxx_kernel = toCuda(torch.Tensor([1, -2, 1]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
def dxx(v, boundary_type='periodic'):
    """x방향 2차 미분"""
    if boundary_type == 'periodic':
        v_padded = F.pad(v, (1, 1, 0, 0), mode='circular')
    else:
        v_padded = F.pad(v, (1, 1, 0, 0), mode='replicate')
    return F.conv2d(v_padded, dxx_kernel, padding=0)

# 2. y방향 2차 미분 (∂²/∂y²)
dyy_kernel = toCuda(torch.Tensor([1, -2, 1]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
def dyy(v, boundary_type='periodic'):
    """y방향 2차 미분"""
    if boundary_type == 'periodic':
        v_padded = F.pad(v, (0, 0, 1, 1), mode='circular')
    else:
        v_padded = F.pad(v, (0, 0, 1, 1), mode='replicate')
    return F.conv2d(v_padded, dyy_kernel, padding=0)

# 3. 교차 미분 (∂²/∂x∂y)
def dxy(v, boundary_type='periodic'):
    """교차 미분 (∂²/∂x∂y = ∂²/∂y∂x)"""
    return dx(dy(v, boundary_type), boundary_type)

# 4. Laplacian (∇² = ∂²/∂x² + ∂²/∂y²)
isotropic_laplacian_kernel = toCuda(torch.Tensor([[0.25, 0.5, 0.25],
                                                 [0.5, -3, 0.5],
                                                 [0.25, 0.5, 0.25]]).unsqueeze(0).unsqueeze(1))
def laplacian(v, boundary_type='periodic'):
    """Laplacian 연산자"""
    if boundary_type == 'periodic':
        v_padded = F.pad(v, (1, 1, 1, 1), mode='circular')
    else:
        v_padded = F.pad(v, (1, 1, 1, 1), mode='replicate')
    return F.conv2d(v_padded, isotropic_laplacian_kernel, padding=0)