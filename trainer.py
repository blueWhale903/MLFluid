import torch 
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim

def fluid_simulation_loss(predicted, target):
    """
    Composite loss function for fluid simulation
    """
    # Standard MSE loss
    velocity_loss = F.mse_loss(predicted, target)
    
    # Gradient consistency loss with higher weight
    def compute_gradients(tensor):
        return [
            torch.gradient(tensor)[0],  # x-gradient
            torch.gradient(tensor)[1]   # y-gradient
        ]
    
    pred_grads = compute_gradients(predicted)
    target_grads = compute_gradients(target)
    
    gradient_loss = sum([F.mse_loss(pg, tg) for pg, tg in zip(pred_grads, target_grads)]) / len(pred_grads)
    
    # Divergence loss with physics constraints
    def divergence_loss(velocity_field):
        # Compute velocity divergence
        div_x = torch.gradient(velocity_field[0])[0]
        div_y = torch.gradient(velocity_field[1])[1]
        return torch.mean((div_x + div_y)**2)
    
    div_loss = divergence_loss(predicted) / 10
    
    # Composite loss with adaptive weighting
    total_loss = (
        velocity_loss + 
        0.2 * gradient_loss + 
        0.1 * div_loss
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # Reduce LR every 3 epochs

    
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
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward Pass
            predictions = model(input_seq)
            predictions = predictions.to(device)

            # Compute Loss
            # predictions = predictions[:, :, -1, :, :]  # Select last time step
            loss = fluid_simulation_loss(predictions, target_seq)
            
            # Backward Pass
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer Step
            optimizer.step()
            
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
        scheduler.step(avg_epoch_loss)
        
        # Epoch progress summary
        if verbose:
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model
