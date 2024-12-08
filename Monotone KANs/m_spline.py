import torch


def M_batch(x, grid, k, device='cpu'):
    """
    Input :
        x : a 1D torch.tensor of shape (num_samples)
        grid : a 1D torch.tensor of shape (num_grid_points)
        k : an int representing the spline order
    Output :
        M-spline values of shape (num_samples, num_splines)
    """
    x= x.to(device)
    grid= grid.to(device)

    grid_ = grid.unsqueeze(dim=0)
    x_ = x.unsqueeze(dim=1)

    # Sort the x_ tensor before proceeding
    x_, indices = torch.sort(x_)
    
    epsilon = 1e-8  # Epsilon to avoid division by zero

    if k == 1:
        # Base case: Piecewise constant, normalized by denom
        denom = (grid_[:, 1:] - grid_[:, :-1])
        
        # Handling any NaN values or division by zero
        if torch.any(torch.isnan(denom)) or torch.any(torch.isinf(denom)) or torch.any(denom == 0):
            denom = torch.where(denom == 0, epsilon, denom)  
            
        value = (x_ >= grid_[:, :-1]) * (x_ < grid_[:, 1:]) / denom
        
        
        # Handling NaNs in value
        value = torch.nan_to_num(value, nan=0.0)


    elif k > 1:

        # Recursive case: M-splines of degree k-1
        Mkm1 = M_batch(x, grid, k-1, device=device)
        denom1 = (k-1)*(grid_[:, k:] - grid_[:, :-k])
        
        # Handling any NaN values or division by zero
        if torch.any(torch.isnan(denom1)) or torch.any(torch.isinf(denom1)) or torch.any(denom1 == 0):
            denom1 = torch.where(denom1 == 0, epsilon, denom1)  

        # Compute M-splines recursively
        
        value = k * ((x_ - grid_[:, :-k]) * Mkm1[:, :-1] + (grid_[:, k:] - x_) * Mkm1[:, 1:]) / denom1
        value = torch.nan_to_num(value, nan=0.0)
        
    
    return value