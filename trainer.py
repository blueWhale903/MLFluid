import torch 
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim

def fluid_simulation_loss(predicted, target, prev_velocity):
    """
    Composite loss function for fluid simulation
    """
    # Standard MSE loss
    velocity_loss = F.mse_loss(predicted, target)
    
    def advection_consistency_loss(pred_velocity, prev_velocity, dt=1.0):
        """Check if the prediction follows the advection equation.
        
        This loss ensures that the velocity field is consistent with the advection equation:
        dv/dt + (v ⋅ ∇)v = 0
        
        Args:
            pred_velocity: Predicted velocity field [batch_size, 2, height, width]
            prev_velocity: Previous velocity field [batch_size, 2, height, width]
            dt: Time step size
            
        Returns:
            Advection consistency loss
        """
        batch_size, _, height, width = pred_velocity.shape
        
        # Calculate dv/dt
        dv_dt = (pred_velocity - prev_velocity) / dt
        
        # Calculate (v ⋅ ∇)v
        # This is a simplified version, as the full calculation is complex
        # We'll approximate it using first-order finite differences
        
        # Extract u and v components
        u = prev_velocity[:, 0]  # x-velocity
        v = prev_velocity[:, 1]  # y-velocity
        
        # Calculate spatial derivatives
        # du/dx
        du_dx = torch.zeros_like(u)
        du_dx[:, 1:-1, 1:-1] = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / 2.0
        
        # du/dy
        du_dy = torch.zeros_like(u)
        du_dy[:, 1:-1, 1:-1] = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / 2.0
        
        # dv/dx
        dv_dx = torch.zeros_like(v)
        dv_dx[:, 1:-1, 1:-1] = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / 2.0
        
        # dv/dy
        dv_dy = torch.zeros_like(v)
        dv_dy[:, 1:-1, 1:-1] = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / 2.0
        
        # Calculate (v ⋅ ∇)v
        # For u component: u * du/dx + v * du/dy
        advection_u = u * du_dx + v * du_dy
        
        # For v component: u * dv/dx + v * dv/dy
        advection_v = u * dv_dx + v * dv_dy
        
        # Stack to get the full advection term
        advection_term = torch.stack([advection_u, advection_v], dim=1)
        
        # The advection equation states: dv/dt + (v ⋅ ∇)v = 0
        # So we compute the residual: dv/dt + (v ⋅ ∇)v
        residual = dv_dt + advection_term
        
        # Return mean squared residual as the loss
        return torch.mean(residual**2)

    def kinetic_energy_loss(pred_velocity, target_velocity):
        """Calculate the difference in kinetic energy between predicted and target velocity fields.
        
        Args:
            pred_velocity: Predicted velocity field [batch_size, 2, height, width]
            target_velocity: Target velocity field [batch_size, 2, height, width]
            
        Returns:
            Kinetic energy difference loss
        """
        # Calculate kinetic energy: 0.5 * (u^2 + v^2)
        pred_energy = 0.5 * (pred_velocity[:, 0]**2 + pred_velocity[:, 1]**2)
        target_energy = 0.5 * (target_velocity[:, 0]**2 + target_velocity[:, 1]**2)
        
        # Calculate the mean squared difference in energy
        energy_diff = pred_energy - target_energy
        return torch.mean(energy_diff**2)
    
    def vorticity_loss(pred_velocity, target_velocity):
        """Calculate the difference in vorticity between predicted and target velocity fields.
        
        Vorticity is defined as the curl of velocity: ω = ∇ × v = dv/dx - du/dy
        
        Args:
            pred_velocity: Predicted velocity field [batch_size, 2, height, width]
            target_velocity: Target velocity field [batch_size, 2, height, width]
            
        Returns:
            Vorticity difference loss
        """
        batch_size, _, height, width = pred_velocity.shape
        
        # Calculate vorticity for predicted velocity
        pred_u = pred_velocity[:, 0]
        pred_v = pred_velocity[:, 1]
        
        pred_du_dy = torch.zeros_like(pred_u)
        pred_du_dy[:, 1:-1, 1:-1] = (pred_u[:, 2:, 1:-1] - pred_u[:, :-2, 1:-1]) / 2.0
        
        pred_dv_dx = torch.zeros_like(pred_v)
        pred_dv_dx[:, 1:-1, 1:-1] = (pred_v[:, 1:-1, 2:] - pred_v[:, 1:-1, :-2]) / 2.0
        
        pred_vorticity = pred_dv_dx - pred_du_dy
        
        # Calculate vorticity for target velocity
        target_u = target_velocity[:, 0]
        target_v = target_velocity[:, 1]
        
        target_du_dy = torch.zeros_like(target_u)
        target_du_dy[:, 1:-1, 1:-1] = (target_u[:, 2:, 1:-1] - target_u[:, :-2, 1:-1]) / 2.0
        
        target_dv_dx = torch.zeros_like(target_v)
        target_dv_dx[:, 1:-1, 1:-1] = (target_v[:, 1:-1, 2:] - target_v[:, 1:-1, :-2]) / 2.0
        
        target_vorticity = target_dv_dx - target_du_dy
        
        # Calculate the mean squared difference in vorticity
        vorticity_diff = pred_vorticity - target_vorticity
        return torch.mean(vorticity_diff**2)
    
    def divergence_loss(velocity_field):
        """Calculate the divergence of the velocity field.
        
        Args:
            velocity_field: Tensor of shape [batch_size, 2, height, width]
                            Channel 0 is u (x-velocity), Channel 1 is v (y-velocity)
        
        Returns:
            Divergence loss (mean squared divergence)
        """
        batch_size, _, height, width = velocity_field.shape
        
        # Calculate divergence: du/dx + dv/dy
        # Use central differences for interior points
        du_dx = torch.zeros_like(velocity_field[:, 0])
        dv_dy = torch.zeros_like(velocity_field[:, 1])
        
        # du/dx using central difference
        du_dx[:, 1:-1, 1:-1] = (velocity_field[:, 0, 1:-1, 2:] - velocity_field[:, 0, 1:-1, :-2]) / 2.0
        
        # Handle boundaries with forward/backward differences
        du_dx[:, 1:-1, 0] = velocity_field[:, 0, 1:-1, 1] - velocity_field[:, 0, 1:-1, 0]
        du_dx[:, 1:-1, -1] = velocity_field[:, 0, 1:-1, -1] - velocity_field[:, 0, 1:-1, -2]
        
        # dv/dy using central difference
        dv_dy[:, 1:-1, 1:-1] = (velocity_field[:, 1, 2:, 1:-1] - velocity_field[:, 1, :-2, 1:-1]) / 2.0
        
        # Handle boundaries with forward/backward differences
        dv_dy[:, 0, 1:-1] = velocity_field[:, 1, 1, 1:-1] - velocity_field[:, 1, 0, 1:-1]
        dv_dy[:, -1, 1:-1] = velocity_field[:, 1, -1, 1:-1] - velocity_field[:, 1, -2, 1:-1]
        
        # Calculate divergence
        divergence = du_dx + dv_dy
        
        # Return mean squared divergence as the loss
        return torch.mean(divergence**2)

    enegy_loss = kinetic_energy_loss(predicted, target)
    vort_loss = vorticity_loss(predicted, target)
    div_loss = divergence_loss(predicted)
    advection_loss = advection_consistency_loss(predicted, prev_velocity)

    # Composite loss with adaptive weighting
    total_loss = (
        velocity_loss + 
        0.1 * div_loss +
        0.05 * vort_loss +
        0.01 * enegy_loss +
        0.05 * advection_loss
    )
    
    return total_loss

def train_fluid_simulation_model(
    model, 
    dataloader, 
    epochs=10, 
    lr=0.001, 
    device=None,
    verbose=True
):
    # Automatic device selection
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Explicitly use "cuda:0"
    
    # Move model to device
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Batch progress bar
        batch_iterator = tqdm(
            enumerate(dataloader), 
            total=len(dataloader), 
            desc=f"Epoch {epoch+1}/{epochs}", 
            leave=True,
            disable=not verbose
        )
        
        for batch_idx, (input_seq, target_seq) in batch_iterator:
            # Move data to device
            input_seq, target_seq = input_seq.to(device, non_blocking=True), target_seq.to(device, non_blocking=True)
            
            prev_velocity = torch.zeros_like(target_seq) if batch_idx == 0 else prev_velocity[:target_seq.size(0)]

            # Zero gradients
            optimizer.zero_grad()
            
            # Forward Pass
            predictions = model(input_seq)
            predictions = predictions.to(device)

            # Compute Loss
            # predictions = predictions[:, :, -1, :, :]  # Select last time step
            loss = fluid_simulation_loss(predictions, target_seq, prev_velocity)
            
            # Backward Pass
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer Step
            optimizer.step()
            
            prev_velocity = predictions.detach()  # Detach to avoid gradient computation

            # Accumulate loss
            total_loss += loss.item()
            
            # Update batch progress bar
            batch_iterator.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Avg Epoch Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # Compute average epoch loss
        avg_epoch_loss = total_loss / len(dataloader)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Epoch progress summary
        if verbose:
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model
