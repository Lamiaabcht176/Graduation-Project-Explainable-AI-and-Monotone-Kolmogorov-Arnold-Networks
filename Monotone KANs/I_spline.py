import torch
from m_spline import M_batch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
cvxopt.solvers.options['show_progress'] = False



# Function to find the max index such that grid <= x for each x
def find_max_index(grid, x, device='cpu'):
      
    mask = torch.where(grid <= x, 0, 1).to(device)
    index = torch.argmax(mask) -1
    # if no element satisfies the conditions, return the last index of grid
    return torch.where(index == -1, grid.shape[0] - 1, index)

  
def I_batch(x, grid, k, extend=True, device='cpu'):
    """
    Input :
        x : a 2D torch.tensor of shape (num_samples, num_splines)
        grid : a 2D torch.tensor of shape (num_splines, num_grid_points)
        k  : an int representing the spline order
    
    Output : 
        I-spline values : a 3D torch.tensor of shape (num_samples, num_splines, G+k) : G: the number of grid intervals, k: spline order.
    """
    x= x.to(device)
    grid= grid.to(device)
    
    num_samples = x.shape[0]  # Number of samples
    num_splines = x.shape[1]  # Number of splines
    G = grid.shape[1] - 1     # Number of grid intervals 

    if extend == True:
      grid = extend_grid(grid, k_extend=k, device=device)

    # Initialize value tensor with zeros
    value = torch.zeros(num_samples, num_splines, G + k, device=device)
    
  
    for i in range(num_splines):  # Loop over splines
        x_spline = x[:, i]  # Samples for the i-th spline
        grid_spline = grid[i, :]
                
        results= []
        # Loop over samples
        for idx, a in enumerate(x_spline):  # Track values in x_spline and their indices
            result= find_max_index(grid_spline, a, device=device)
            results.append(result)
            j_list = torch.stack(results)
            
            j = j_list[idx].item()  # Get the corresponding `j` index for the current sample
            # Case 2: I(x) = 1 if i < j - k + 1
            if i < j - k + 1:
                value[idx, i, :] = 1  # Broadcast 1 across the (G + k) dimension

            # Case 3: recursive formula if j - k + 1 <= i <= j
            if j - k + 1 <= i <= j:
                # Calculate M-splines for k+1 order (calling the m_spline() function)
                Mkp1 = M_batch(x_spline, grid_spline, k + 1, device=device)
                t_mkp1 = grid_spline[min(i + k +1, grid_spline.shape[0] - 1): min(j + k + 2, grid_spline.shape[0])]
                t_m = grid_spline[i : j + 1]
                # Calculate the sum
                value[idx, i, :] = torch.sum(((t_mkp1 - t_m) * Mkp1[idx, i:j+1] / (k + 1)), dim=0)
                
            if a==1 :
                value[idx, i, :] = 1
            
                  
    value = torch.nan_to_num(value, nan=0.0)


    return value

# Function that wraps I_batch() to handle constant splines
def get_I_batch(x, grid, k, extend=False, device='cpu'):
  single_grid = grid[0,:]
  grid = single_grid.repeat(grid.shape[0] + 1, 1)
  x = torch.cat((x[:,0].unsqueeze(dim=1), x), dim=1)
  result = I_batch(x, grid, k, extend=extend, device=device)
  return result[:, 1:, :]


def coef2curve_mono(x_eval, grid, coef, k, extend=True, device="cpu"):
    """
    converts I-spline coefficients to I-spline curves. Evaluates x on I-spline curves.
    Input :
        x_eval: (num_samples, num_splines)
        grid: (num_splines, num_grid_points) 
        coef: (num_splines, out_dim, G+k)

    Output : 
        y_eval : 3D torch tensor with shape: (num_samples, num_splines, out_dim)
    """
    x_eval = x_eval.to(device)
    grid= grid.to(device)
    coef = coef.to(device)
    
    x_eval, n = torch.sort(x_eval, dim=0)

    i_values = get_I_batch(x_eval, grid, k=k, extend=False, device=device)

    y_eval = torch.einsum('ijk,jlk->ijl', i_values, coef.to(i_values.dtype).to(i_values.device))
    
    return y_eval



def curve2coef_mono(x_eval, y_eval, grid, k, lamb=1e-3, extend=True, device='cpu'):
    """
    Convert I-spline curves to I-spline coefficients solving least squares with non-negativity constraints 
    using matrix inversion and thresholding.

    Input:
        x_eval : 2D torch.tensor of shape (num_samples, num_splines)
        y_eval : 3D torch.tensor of shape (num_samples, num_splines, out_dim)
        grid : 2D torch.tensor of shape (num_splines, num_grid_points)
        k : int, spline order
        lamb : float, regularized least square lambda

    Output:
        coef : 3D torch.tensor of shape (num_splines, out_dim, G+k)
    """
    num_samples = x_eval.shape[0]
    num_splines = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] + k - 1
    
    x_eval, n = torch.sort(x_eval, dim=0)  
    
    
    # I-spline basis functions matrix
    mat = get_I_batch(x_eval, grid, k, extend=False, device=device)  # shape: (num_samples, num_splines, G+k)
    
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(num_splines, out_dim, num_samples, n_coef)  # shape: (num_splines, out_dim, num_samples, n_coef)

    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)  # shape: (num_splines, out_dim, num_samples, 1)
    device = mat.device

    # Construct A = mat and B = y_eval
    A = mat 
    B = y_eval.to(A.device)

    # Compute A^T A and A^T B
    AtA = torch.einsum('ijmn,ijnp->ijmp', A.permute(0, 1, 3, 2), A) #shape (num_splines, out_dim, n_coef, n_coef)
    AtB = torch.einsum('ijmn,ijnp->ijmp', A.permute(0, 1, 3, 2), B) #shape (num_splines, out_dim, n_coef, 1)

    # Identity matrix for regularization term
    n1, n2, n = AtA.shape[0], AtA.shape[1], AtA.shape[2]
    identity = torch.eye(n,n)[None, None, :, :].expand(n1, n2, n, n).to(A.device)
    
    # Add regularization to AtA
    Q = AtA + lamb * identity  # Q = (A^T A + lambda*I)
    c = AtB   # c = A^T B 

    # Initialize coefficients
    coef = torch.zeros(num_splines, out_dim, n_coef, device=device)

    # Iterate over splines and output dimensions
    for i in range(num_splines):
        for j in range(out_dim):
            
            Q_j = Q[i, j]  # (n_coef, n_coef)
            c_j = c[i, j]  # (n_coef, 1)

            # Invert the matrix Q_j (regularized matrix)
            Q_j_inv = torch.linalg.inv(Q_j)  # Inverse of Q_j
            coef_j =  Q_j_inv @ c_j# Solution: coef_j = inv(Q_j) * c_j

            # Apply threshold to ensure non-negativity
            coef_j = torch.clamp(coef_j, min=0)  
            coef[i, j] = coef_j.squeeze(-1)  # Remove last dimension

    return coef


  

def extend_grid(grid, k_extend=0, device='cpu'):
    
    grid = grid.to(device)
    grid_start = grid[:, :1].repeat(1, k_extend)  # Repeating the first column k_extend times at the beggining
    grid_end = grid[:, -1:].repeat(1, k_extend)  # Repeating the last column k_extend times at the end

    # Concatenate the repeated values at the beginning and end with the original grid
    extended_grid = torch.cat([grid_start, grid, grid_end], dim=1)

    return extended_grid