import torch
import torch.nn as nn
import numpy as np
from I_spline import *
from utils import sparse_mask
import torch.nn.init as init

class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        size: int
            the number of splines = input dimension * output dimension
        k: int
            the piecewise polynomial order of splines
        grid: 2D torch.float
            grid points
        noises: 2D torch.float
            injected noises to splines at initialization (to break degeneracy)
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base: 1D torch.float
            magnitude of the residual function b(x)
        scale_sp: 1D torch.float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
        weight_sharing: 1D tensor int
            allow spline activations to share parameters
        lock_counter: int
            counter how many activation functions are locked (weight sharing)
        lock_id: 1D torch.int
            the id of activation functions that are locked
        device: str
            device
    
    Methods:
    --------
        __init__():
            initialize a KANLayer
        forward():
            forward 
        update_grid_from_samples():
            update grids based on samples' incoming activations
        initialize_grid_from_parent():
            initialize grids from another model
        get_subset():
            get subset of the KANLayer (used for pruning)
        lock():
            lock several activation functions to share parameters
        unlock():
            unlock already locked activation functions
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, base_fun=torch.nn.Sigmoid(), grid_eps=0.02, grid_range=[0, 1], sp_trainable=True, sb_trainable=True, save_plot_data = True, device='cpu', sparse_init=False):
        ''''
        initialize a KANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base : float
                the scale of the residual function b(x). Default: 1.0.
            scale_sp : float
                the scale of the base function spline(x). Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        (3, 5)
        '''
        super(KANLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k
        self.device = device

        # shape: (size, num)
        ### grid size: (batch, in_dim, out_dim, G + 1) => (batch, in_dim, G + 2*k + 1)
        
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        grid = extend_grid(grid, k_extend=k, device=device)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        scale = 0.01
        #self.coef = nn.Parameter(torch.abs(init.xavier_normal_(torch.empty(in_dim, out_dim, num + 3 * k))* scale))
        self.coef = nn.Parameter(torch.abs(torch.normal(0, 1, size=(in_dim, out_dim, num + 3 * k))))

        #if isinstance(scale_base, float):
        if sparse_init:
            mask = sparse_mask(in_dim, out_dim)
        else:
            mask = 1.
        
        #self.scale_base = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_base * mask).requires_grad_(sb_trainable)  # make scale trainable
        
        #self.scale_base= self.scale_base.to(device)
        
        #self.scale_base = torch.nn.Parameter(scale_base.to(device)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * mask).requires_grad_(sp_trainable)  # make scale trainable
        #self.scale_sp= self.scale_sp.to(device)

        self.base_fun = base_fun
        

        self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        self.grid_eps = grid_eps
        
        ### remove weight_sharing & lock parts
        #self.weight_sharing = torch.arange(out_dim*in_dim).reshape(out_dim, in_dim)
        #self.lock_counter = 0
        #self.lock_id = torch.zeros(out_dim*in_dim).reshape(out_dim, in_dim)
        self.device=device

    def forward(self, x):
        '''
        KANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of samples, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs
        '''
        x = x.to(self.device)
        batch = x.shape[0]
        
        # ordering x values
        sorted_indices = torch.argsort(x, dim=0)  
        x_sorted = torch.gather(x, dim=0, index=sorted_indices)  
        
        preacts = x_sorted[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)
        base = self.base_fun(x).to(self.device)  # (batch, in_dim)
        
        y_sorted = coef2curve_mono(x_eval=x_sorted, grid=self.grid, coef=self.coef, k=self.k, device=self.device)  # shape: (batch, in_dim, out_dim)
        #reordering y values to match the ordering of
        reordered_y = torch.zeros_like(y_sorted).to(self.device)

        for i in range(batch):
            reordered_y[sorted_indices[i], :, :] = y_sorted[i, :, :] 
        
        postspline = reordered_y.clone().permute(0, 2, 1)  # shape: (batch, out_dim, in_dim)
        
        # Ensure scale_sp is non-negative
        self.scale_sp.data = torch.abs(self.scale_sp.data)
        
        y = self.mask[None, :, :].to(self.device) * reordered_y
        
        postacts = y.clone().permute(0, 2, 1).to(self.device)
        
        y = torch.sum(y, dim=1).to(self.device)  # shape (batch, out_dim)

        return y, preacts, postacts, postspline


    def update_grid_from_samples(self, x, mode='sample'):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        '''
        x = x.to(self.device)
        batch = x.shape[0]
        #x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        # ordering x values 
        original_indices = torch.argsort(x, dim=0)
        sorted_x = torch.gather(x, dim=0, index=original_indices)
        
        y_eval = coef2curve_mono(sorted_x, self.grid, self.coef, self.k, device=self.device)
        indices_expanded = original_indices.unsqueeze(2).expand(-1, -1, y_eval.shape[2])
    
        y_eval = torch.gather(y_eval, dim=0, index=indices_expanded)
        
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = sorted_x[ids, :].permute(1,0).to(self.device)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid.to(self.device)
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0).to(self.device)
            y_eval = coef2curve_mono(x_pos, self.grid, self.coef, self.k, device=self.device)
        
        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef_mono(sorted_x, y_eval, self.grid, self.k, device=self.device)

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        '''
        update grid from a parent KANLayer & samples
        
        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-1.0000, -0.8000, -0.6000, -0.4000, -0.2000,  0.0000,  0.2000,  0.4000,
          0.6000,  0.8000,  1.0000]])
        '''
        
        batch = x.shape[0]
        x = x.to(self.device)
        parent = parent.to(self.device)
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve_mono(x_pos, parent.grid, parent.coef, parent.k, device=self.device)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve_mono(x_pos, parent.grid, parent.coef, parent.k, device=self.device)
        
        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef_mono(x_pos, y_eval, self.grid, self.k, device=self.device)

    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : KANLayer
            
        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:,out_id]
        spb.scale_base.data = self.scale_base[in_id][:,out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:,out_id]
        spb.mask.data = self.mask[in_id][:,out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb
    
    
    def swap(self, i1, i2, mode='in'):
        
        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)

