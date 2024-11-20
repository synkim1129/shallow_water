import torch
from derivatives import dx, dy

def compute_boundary_loss(h_delta, u, v, boundary_mask, boundary_type, H0=100.0):
    """
    Compute loss terms for enforcing boundary conditions
    """
    if boundary_type == 'rigid':
        # 강체벽 경계조건: 경계에서 속도가 0이어야 함
        velocity_loss = torch.mean((u**2 + v**2) * boundary_mask, dim=(1, 2, 3))
        return velocity_loss

    elif boundary_type == 'outflow':
        # 유출 경계조건: 경계에서 수직방향 gradient가 0이어야 함
        h_gradient = torch.abs(dx(h_delta)) * boundary_mask[:,:,:,1:-1] + \
                    torch.abs(dy(h_delta)) * boundary_mask[:,:,1:-1,:]
        u_gradient = torch.abs(dx(u)) * boundary_mask[:,:,:,1:-1] + \
                    torch.abs(dy(u)) * boundary_mask[:,:,1:-1,:]
        v_gradient = torch.abs(dx(v)) * boundary_mask[:,:,:,1:-1] + \
                    torch.abs(dy(v)) * boundary_mask[:,:,1:-1,:]
        
        return torch.mean(h_gradient + u_gradient + v_gradient, dim=(1, 2, 3))

    elif boundary_type == 'reflecting':
        # 반사 경계조건: 경계에서 수직방향 속도가 0이어야 함
        horizontal_boundary = boundary_mask[:,:,0,:] + boundary_mask[:,:,-1,:]
        vertical_boundary = boundary_mask[:,:,:,0] + boundary_mask[:,:,:,-1]
        
        velocity_loss = torch.mean(
            (u**2) * vertical_boundary.unsqueeze(-1) +
            (v**2) * horizontal_boundary.unsqueeze(-2),
            dim=(1, 2, 3)
        )
        
        return velocity_loss

    elif boundary_type == 'periodic':
        # 주기적 경계조건: 경계에서의 값들이 반대쪽 경계와 같아야 함
        h_loss_x = torch.mean((h_delta[:,:,:,0] - h_delta[:,:,:,-1])**2, dim=(1, 2))
        h_loss_y = torch.mean((h_delta[:,:,0,:] - h_delta[:,:,-1,:])**2, dim=(1, 2))
        
        u_loss_x = torch.mean((u[:,:,:,0] - u[:,:,:,-1])**2, dim=(1, 2))
        u_loss_y = torch.mean((u[:,:,0,:] - u[:,:,-1,:])**2, dim=(1, 2))
        
        v_loss_x = torch.mean((v[:,:,:,0] - v[:,:,:,-1])**2, dim=(1, 2))
        v_loss_y = torch.mean((v[:,:,0,:] - v[:,:,-1,:])**2, dim=(1, 2))
        
        return h_loss_x + h_loss_y + u_loss_x + u_loss_y + v_loss_x + v_loss_y

    else:
        raise ValueError(f"Unknown boundary type: {boundary_type}")