import torch
import numpy as np

class WaterSurfaceDataset:
    def __init__(self, width, height, batch_size=100, dataset_size=1000, 
                 average_sequence_length=5000, H0=100.0, dt=1.0,
                 boundary_type='rigid'):
        """
        Initialize water surface dataset
        Args:
            width: width of domain
            height: height of domain
            batch_size: batch size for ask()
            dataset_size: size of dataset
            average_sequence_length: average length of sequence until domain gets reset
            H0: mean water depth
            dt: timestep of simulation
            boundary_type: type of boundary condition ('rigid', 'outflow', 'reflecting')
        """
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.average_sequence_length = average_sequence_length
        self.H0 = torch.tensor(H0)
        self.dt = dt
        self.g = torch.tensor(9.81)
        self.boundary_type = boundary_type
        
        # Initialize state variables
        # h_delta: deviation from mean water depth
        self.h_delta = torch.zeros(dataset_size, 1, height, width)
        self.u = torch.zeros(dataset_size, 1, height, width)
        self.v = torch.zeros(dataset_size, 1, height, width)
        
        # Create boundary mask based on boundary type
        self.boundary_mask = self.create_boundary_mask()
        
        self.t = 0
        self.i = 0
        
        self.env_info = [{} for _ in range(dataset_size)]
        
        # Initialize all environments
        for i in range(dataset_size):
            self.reset_env(i)

    def create_boundary_mask(self):
        """
        Create boundary mask based on the boundary type
        0: interior
        1: boundary
        """
        mask = torch.zeros(1, self.height, self.width)
        
        if self.boundary_type in ['rigid', 'reflecting', 'outflow']:
            # Set all boundaries to 1
            mask[0, 0, :] = 1  # top
            mask[0, -1, :] = 1  # bottom
            mask[0, :, 0] = 1  # left
            mask[0, :, -1] = 1  # right
        elif self.boundary_type == 'periodic':
            # periodic 경계조건에서는 경계 마스크가 필요 없음
            pass
        
        return mask

    def reset_env(self, index):
        """Reset environment[index] to a new, randomly chosen state"""
        # Initialize base state (deviations from mean depth)
        h_delta = torch.zeros(1, self.height, self.width)
        u = torch.zeros(1, self.height, self.width)
        v = torch.zeros(1, self.height, self.width)
        
        c = torch.sqrt(self.g * self.H0)  # characteristic wave speed
        
        # Random velocity component base strength (as fraction of wave speed)
        velocity_strength = 0.1  # 10% of wave speed
        base_random_u = torch.tensor(np.random.uniform(-1, 1, u.shape)) * velocity_strength * c
        base_random_v = torch.tensor(np.random.uniform(-1, 1, v.shape)) * velocity_strength * c

        pattern = np.random.choice([
            "standing_waves",
            "multiple_peaks",
            "wave_train",
        ])

        if pattern == "standing_waves":
            kx = torch.tensor(np.random.uniform(1, 5))
            ky = torch.tensor(np.random.uniform(1, 5))
            A = torch.tensor(np.random.uniform(0.01, 0.03) * self.H0)
            
            y = torch.arange(self.height)
            x = torch.arange(self.width)
            Y, X = torch.meshgrid(y, x, indexing='ij')
            h_delta = A * torch.sin(kx*X) * torch.sin(ky*Y)
            
            u = A * c * kx * torch.cos(kx*X) * torch.sin(ky*Y) + base_random_u
            v = A * c * ky * torch.sin(kx*X) * torch.cos(ky*Y) + base_random_v
            
            self.env_info[index].update({
                "type": pattern,
                "kx": kx,
                "ky": ky,
                "amplitude": A
            })

        elif pattern == "multiple_peaks":
            n_peaks = np.random.randint(3, 8)
            peaks_info = []
            for _ in range(n_peaks):
                x0 = torch.tensor(np.random.uniform(0, self.width))
                y0 = torch.tensor(np.random.uniform(0, self.height))
                sigma = torch.tensor(np.random.uniform(5, 20))
                A = torch.tensor(np.random.uniform(0.01, 0.03) * self.H0)
                
                y = torch.arange(self.height)
                x = torch.arange(self.width)
                Y, X = torch.meshgrid(y, x, indexing='ij')
                h_delta += A * torch.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2))
                
                # Add radial velocity for each peak
                r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
                r = torch.clamp(r, min=1.0)  # Prevent division by zero
                v_r = A * c * torch.exp(-r**2/(2*sigma**2))
                u += v_r * (X - x0) / r
                v += v_r * (Y - y0) / r
                
                peaks_info.append({"x0": x0, "y0": y0, "sigma": sigma, "A": A})
                
            u += base_random_u
            v += base_random_v
            
            self.env_info[index].update({
                "type": pattern,
                "peaks": peaks_info
            })

        elif pattern == "wave_train":
            k = torch.tensor(np.random.uniform(2, 6))
            theta = torch.tensor(np.random.uniform(0, 2*np.pi))
            A = torch.tensor(np.random.uniform(0.01, 0.03) * self.H0)
            
            y = torch.arange(self.height)
            x = torch.arange(self.width)
            Y, X = torch.meshgrid(y, x, indexing='ij')
            phase = k * (X*torch.cos(theta) + Y*torch.sin(theta))
            h_delta += A * torch.sin(phase)
            
            # Set velocity for wave propagation
            u = A * c * k * torch.cos(theta) * torch.cos(phase) + base_random_u
            v = A * c * k * torch.sin(theta) * torch.cos(phase) + base_random_v
            
            self.env_info[index].update({
                "type": pattern,
                "wavenumber": k,
                "direction": theta,
                "amplitude": A
            })

        # Apply boundary conditions to initial state
        if self.boundary_type == 'rigid':
            # Set velocities to zero at boundaries
            u = u * (1 - self.boundary_mask)
            v = v * (1 - self.boundary_mask)
        elif self.boundary_type == 'reflecting':
            # Reflect velocities at boundaries
            u = u * (1 - 2 * self.boundary_mask)
            v = v * (1 - 2 * self.boundary_mask)
        # For outflow, we don't modify the velocities at boundaries

        # Update state
        self.h_delta[index] = h_delta
        self.u[index] = u
        self.v[index] = v

    def ask(self, debug=False):
        """
        Ask for a batch of initial conditions
        Returns:
            h_delta: deviation from mean water depth
            u: x-velocity
            v: y-velocity
            boundary_mask: boundary condition mask
        """
        self.indices = np.random.choice(self.dataset_size, self.batch_size)
        if debug:
            return (self.h_delta[self.indices], 
                    self.u[self.indices], 
                    self.v[self.indices],
                    self.boundary_mask.expand(self.batch_size, -1, -1, -1),
                    [self.env_info[i] for i in self.indices])
        else:
            return (self.h_delta[self.indices], 
                    self.u[self.indices], 
                    self.v[self.indices],
                    self.boundary_mask.expand(self.batch_size, -1, -1, -1))

    def tell(self, h_delta, u, v):
        """
        Return the updated fluid state to the dataset
        Args:
            h_delta: deviation from mean water depth
            u: updated x-velocity
            v: updated y-velocity
        """
        # Apply boundary conditions
        if self.boundary_type == 'rigid':
            u = u * (1 - self.boundary_mask)
            v = v * (1 - self.boundary_mask)
        elif self.boundary_type == 'reflecting':
            u = u * (1 - 2 * self.boundary_mask)
            v = v * (1 - 2 * self.boundary_mask)
            
        self.h_delta[self.indices] = h_delta.detach()
        self.u[self.indices] = u.detach()
        self.v[self.indices] = v.detach()
        
        self.t += 1
        if self.t % (self.average_sequence_length/self.batch_size) == 0:
            self.reset_env(int(self.i))
            self.i = (self.i+1) % self.dataset_size