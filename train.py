import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from derivatives import dx, dy, laplacian, toCuda, toCpu, params
from setups import WaterSurfaceDataset
from Logger import Logger, t_step
from pde_cnn import WaterSurfaceUNet
from boundary_losses import compute_boundary_loss
from get_param import get_hyperparam

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

# Print parameters
print("\nStarting shallow water simulation training:")
print(f"Physical parameters - g: {params.g}, H0: {params.H0}, dt: {params.dt}")
print(f"Domain size - Width: {params.width}, Height: {params.height}")
print(f"Training parameters - Epochs: {params.n_epochs}, Batches per epoch: {params.n_batches_per_epoch}")
print(f"Dataset size: {params.dataset_size}, Batch size: {params.batch_size}")
print(f"Boundary type: {params.boundary_type}")
print("Loss weights:")
print(f"  Mass conservation: {params.loss_mass}")
print(f"  Momentum conservation: {params.loss_momentum}")
print(f"  Energy conservation: {params.loss_energy}")
print(f"  Positive depth: {params.loss_positive_depth}")
print(f"  CFL condition: {params.loss_cfl}")
print("\nTraining progress:")



# Initialize physical parameters
g = params.g
H0 = params.H0
dt = params.dt

# Initialize fluid model
fluid_model = toCuda(WaterSurfaceUNet(params.hidden_size))
fluid_model.train()

# Initialize Optimizer
optimizer = Adam(fluid_model.parameters(), lr=params.lr)

# Initialize Logger and load model / optimizer if specified
logger = Logger(get_hyperparam(params), use_tensorboard=params.log)
if params.load_latest or params.load_date_time is not None or params.load_index is not None:
    load_logger = Logger(get_hyperparam(params), use_tensorboard=False)
    if params.load_optimizer:
        params.load_date_time, params.load_index = logger.load_state(fluid_model, optimizer, params.load_date_time, params.load_index)
    else:
        params.load_date_time, params.load_index = logger.load_state(fluid_model, None, params.load_date_time, params.load_index)
    params.load_index = int(params.load_index)
    print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index

# Initialize Dataset
dataset = WaterSurfaceDataset(
    width=params.width,
    height=params.height,
    batch_size=params.batch_size,
    dataset_size=params.dataset_size,
    average_sequence_length=params.average_sequence_length,
    H0=params.H0,
    dt=params.dt
)

# training loop
for epoch in range(params.load_index, params.n_epochs):
    for i in range(params.n_batches_per_epoch):
        # Get batch of data
        # import pdb; pdb.set_trace()
        h_delta_old, u_old, v_old, boundary_mask = toCuda(dataset.ask())
        
        # Predict new fluid state
        h_delta_new, u_new, v_new = fluid_model(h_delta_old, u_old, v_old)
        
        # Calculate total height for conservation laws
        h_new = h_delta_new + params.H0
        h_old = h_delta_old + params.H0
        
        # Mass conservation loss
        h_avg = (h_new + h_old) * 0.5
        u_avg = (u_new + u_old) * 0.5
        v_avg = (v_new + v_old) * 0.5
        
        mass_time = (h_new - h_old) / dt
        mass_flux_x = dx(h_avg * u_avg, boundary_type=params.boundary_type)
        mass_flux_y = dy(h_avg * v_avg, boundary_type=params.boundary_type)
        mass = mass_time + mass_flux_x + mass_flux_y
        loss_mass = torch.mean(mass**2, dim=(1, 2, 3))
        
        # 2. Momentum conservation loss
        # x-momentum: ∂(u)/∂t + (g)∂h/∂x + lap(u)  = 0
        # y-momentum: ∂(v)/∂t + (g)∂h/∂y + lap(v)  = 0      
        
        # Momentum conservation loss
        # x-momentum
        mom_x_time = (u_new - u_old) / dt
        mom_x_press = g * dx(h_delta_new, boundary_type=params.boundary_type)
        mom_x_visc = laplacian(u_avg, boundary_type=params.boundary_type)  # 점성항
        mom_x = mom_x_time + mom_x_press - mom_x_visc
        loss_momentum_x = torch.mean(mom_x**2, dim=(1, 2, 3))

        # y-momentum
        mom_y_time = (v_new - v_old) / dt
        mom_y_press = g * dy(h_delta_new, boundary_type=params.boundary_type)
        mom_y_visc = laplacian(v_avg, boundary_type=params.boundary_type)  # 점성항
        mom_y = mom_y_time + mom_y_press - mom_y_visc
        loss_momentum_y = torch.mean(mom_y**2, dim=(1, 2, 3))

        # Total momentum loss
        loss_momentum = loss_momentum_x + loss_momentum_y

                    
        # CFL condition penalty
        # wave_speed = torch.sqrt(g * h_new)
        # velocity_magnitude = torch.sqrt(u_new**2 + v_new**2)
        # cfl = dt * (velocity_magnitude + wave_speed)  # dx=1 assumed
        # loss_cfl = torch.mean(F.relu(cfl - 0.5)**2)  # penalty if CFL > 0.5
        
        # Boundary condition loss
        loss_boundary = compute_boundary_loss(
            h_delta_new, u_new, v_new,
            boundary_mask, dataset.boundary_type,
            H0=params.H0
        )
        
        # Combine losses
        loss = (params.loss_mass * loss_mass + 
                params.loss_momentum * (loss_momentum_x + loss_momentum_y) + 
                params.loss_boundary * loss_boundary)
                # params.loss_positive_depth * loss_positive_depth +
                # params.loss_cfl * loss_cfl)
                
        loss = torch.mean(loss)
        
        # Compute gradients and optimize
        optimizer.zero_grad()
        loss = loss * params.loss_multiplier
        loss.backward()
        
        # Optional: clip gradients
        if params.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(fluid_model.parameters(), params.clip_grad_norm)
        
        optimizer.step()
        
        # Return updated state to dataset
        dataset.tell(toCpu(h_delta_new), toCpu(u_new), toCpu(v_new))
        
        # Log training metrics
        if i % 10 == 0:
            with torch.no_grad():
                log_loss = np.log10(loss.cpu().numpy())
                log_loss_mass = np.log10(torch.mean(loss_mass).cpu().numpy())
                log_loss_momentum = np.log10(torch.mean(loss_momentum_x).cpu().numpy() + torch.mean(loss_momentum_y.cpu()).numpy())
                log_loss_boundary = np.log10(torch.mean(loss_boundary).cpu().numpy())
                
                logger.log(f"loss_total", log_loss, epoch*params.n_batches_per_epoch + i)
                logger.log(f"loss_mass", log_loss_mass, epoch*params.n_batches_per_epoch + i)
                logger.log(f"loss_momentum", log_loss_momentum, epoch*params.n_batches_per_epoch + i)
                
                

                if i % 100 == 0:
                    print(f"Epoch {epoch:3d} | Batch {i:5d}/{params.n_batches_per_epoch} | "
                            f"Log Loss: {log_loss:.3f} | Log Mass Loss: {log_loss_mass:.3f} | "
                            f"Log Momentum Loss: {log_loss_momentum:.3f} | Log Boundary Loss: {log_loss_boundary:.3f}")
                else:
                    print(f"Epoch {epoch:3d} | Batch {i:5d}/{params.n_batches_per_epoch} | "
                            f"Log Loss: {log_loss:.3f} | Log Mass Loss: {log_loss_mass:.3f} | "
                            f"Log Momentum Loss: {log_loss_momentum:.3f} | Log Boundary Loss: {log_loss_boundary:.3f}", end='\r')
    
    # Print epoch summary
    print(f"Completed epoch {epoch:3d}/{params.n_epochs-1} | "
          f"Final Log Loss: {log_loss:.3f} | Final Log Mass Loss: {log_loss_mass:.3f} | "
          f"Final Log Momentum Loss: {log_loss_momentum:.3f} | Final Log Boundary Loss: {log_loss_boundary:.3f}")
    
    # Save state after every epoch
    if params.log:
        logger.save_state(fluid_model, optimizer, epoch+1)
        print(f"Saved model state for epoch {epoch+1}")

print("\nTraining completed!")