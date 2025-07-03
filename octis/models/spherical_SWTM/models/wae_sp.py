import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from octis.models.spherical_SWTM.models.distributions.hyperbolic import *
from octis.models.spherical_SWTM.models.distributions.power_spherical import *
from octis.models.spherical_SWTM.models.distributions.von_mises_fisher import rand_von_mises_fisher

# WAE model
class WAE(nn.Module):
    def __init__(self,
                 encode_dims=[2000, 1024, 512, 20],
                 decode_dims=[20, 1024, 2000],
                 dropout=0.0,
                 nonlin='relu',
                 dist='vmf',
                 batch_size=256,
                 temperature=0.7
                 ):
        super(WAE, self).__init__()
        
        self.dist=dist
        self.batch_size = batch_size

        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encode_dims[i], encode_dims[i+1])
            for i in range(len(encode_dims)-1)
        })

        self.decoder = nn.ModuleDict({
            f'dec_{i}': nn.Linear(decode_dims[i], decode_dims[i+1])
            for i in range(len(decode_dims)-1)
        })
        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[nonlin]
        self.z_dim = encode_dims[-1]
        
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        self.proj = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.LayerNorm(self.z_dim)
        )
    
    def encode(self, x):
        hid = x
        n_layers = len(self.encoder)
        for i, (_, layer) in enumerate(self.encoder.items()):
            if i < n_layers - 1:
                hid = self.dropout(layer(hid))
                hid = self.nonlin(hid)
            else:
                hid = layer(hid)
        return hid


    def decode(self, z):
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i < len(self.decoder)-1:
                hid = self.nonlin(self.dropout(hid))
        return hid
    
    def forward(self, x):
        z = self.encode(x)
        z_norm = F.normalize(z, p=2, dim=1)
        z_proj = self.proj(z_norm)  # Now has richer variation
        theta = F.softmax(self.dropout(z_proj) / self.temperature, dim=1)
        x_reconst = self.decode(theta)
        return x_reconst, z_norm


    def sample(self, dirichlet_alpha=0.1, ori_data=None):
        if self.dist == 'dirichlet':
            z_true = np.random.dirichlet(
                np.ones(self.z_dim)*dirichlet_alpha, size=self.batch_size)
            z_true = torch.from_numpy(z_true).float()
            return z_true
        
        elif self.dist == 'gaussian':
            z_true = np.random.randn(self.batch_size, self.z_dim)
            z_true = torch.softmax(torch.from_numpy(z_true), dim=1).float()
            return z_true
        
        elif self.dist == 'gmm_std':
            odes = np.eye(self.z_dim)*20
            ides = np.random.randint(low=0, high=self.z_dim, size=self.batch_size)
            mus = odes[ides]
            sigmas = np.ones((self.batch_size, self.z_dim))*0.2*20
            z_true = np.random.normal(mus, sigmas)
            z_true = F.softmax(torch.from_numpy(z_true).float(), dim=1)
            return z_true
        
        elif self.dist=='gmm_ctm' and ori_data!=None:
            with torch.no_grad():
                hid_vecs = self.encode(ori_data).cpu().numpy()
                gmm = GaussianMixture(n_components=self.z_dim,covariance_type='full',max_iter=200)
                gmm.fit(hid_vecs)
                #hid_vecs = torch.from_numpy(hid_vecs).to(self.device)
                gmm_spls, _spl_lbls = gmm.sample(n_samples=len(ori_data))
                theta_prior = torch.from_numpy(gmm_spls).float()
                theta_prior = F.softmax(theta_prior,dim=1)
                return theta_prior

        elif self.dist=='vmf':
            mu = torch.eye(self.z_dim, dtype=torch.float)[0]
            kappa = 10
            X = rand_von_mises_fisher(mu, kappa=kappa, N=self.batch_size)
            target_latent = torch.from_numpy(X).to(dtype=torch.float)
            return target_latent
        
        elif self.dist=='mixture_vmf':
            ps = np.ones(2*self.z_dim)/(2*self.z_dim)
            mus = torch.cat((torch.eye(self.z_dim, dtype=torch.float64), -torch.eye(self.z_dim, dtype=torch.float64)), 0)
            mus = F.normalize(mus, p=2, dim=-1)
            Z = np.random.multinomial(self.batch_size, ps)

            nonzero_indices = np.where(Z > 0)[0]
            num_samples_per_component = Z[nonzero_indices]
            selected_mus = mus[nonzero_indices]

            X = torch.tensor([]).to(dtype=torch.float)
            for i, num_samples in enumerate(num_samples_per_component):
                vmf = rand_von_mises_fisher(selected_mus[i], kappa=10, N=int(num_samples))
                vmf = torch.tensor(vmf).to(dtype=torch.float)
                X = torch.cat((X, vmf))  # Use extend for efficiency
            
            return X
        
        elif self.dist == "unif_sphere": ## unif sphere
            target_latent = torch.randn(self.batch_size, self.z_dim)
            target_latent = F.normalize(target_latent, p=2, dim=-1)
            return target_latent
        
        else:
            return self.sample(dist='dirichlet',batch_size=self.batch_size)

    def mmd_loss(self, x, y, device, t=0.1, kernel='diffusion'):
        '''
        computes the mmd loss with information diffusion kernel
        :param x: batch_size * latent dimension
        :param y:
        :param t:
        :return:
        '''
        eps = 1e-6
        n, d = x.shape
        if kernel == 'tv':
            sum_xx = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(i+1, n):
                    sum_xx = sum_xx + torch.norm(x[i]-x[j], p=1).to(device)
            sum_xx = sum_xx / (n * (n-1))

            sum_yy = torch.zeros(1).to(device)
            for i in range(y.shape[0]):
                for j in range(i+1, y.shape[0]):
                    sum_yy = sum_yy + torch.norm(y[i]-y[j], p=1).to(device)
            sum_yy = sum_yy / (y.shape[0] * (y.shape[0]-1))

            sum_xy = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(y.shape[0]):
                    sum_xy = sum_xy + torch.norm(x[i]-y[j], p=1).to(device)
            sum_yy = sum_yy / (n * y.shape[0])
        else:
            qx = torch.sqrt(torch.clamp(x, eps, 1))
            qy = torch.sqrt(torch.clamp(y, eps, 1))
            xx = torch.matmul(qx, qx.t())
            yy = torch.matmul(qy, qy.t())
            xy = torch.matmul(qx, qy.t())

            def diffusion_kernel(a, tmpt, dim):
                return torch.exp(-torch.acos(a).pow(2)) / tmpt

            off_diag = 1 - torch.eye(n).to(device)
            k_xx = diffusion_kernel(torch.clamp(xx, 0, 1-eps), t, d-1)
            k_yy = diffusion_kernel(torch.clamp(yy, 0, 1-eps), t, d-1)
            k_xy = diffusion_kernel(torch.clamp(xy, 0, 1-eps), t, d-1)
            sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
            sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
            sum_xy = 2 * k_xy.sum() / (n * n)
        return sum_xx + sum_yy - sum_xy
    
####################################################################################################################################################################################
    def emd1D(self, u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
        n = u_values.shape[-1]
        m = v_values.shape[-1]

        device = u_values.device
        dtype = u_values.dtype

        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device) if u_weights is None else u_weights
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device) if v_weights is None else v_weights

        if require_sort:
            u_values, u_sorter = torch.sort(u_values, -1)
            v_values, v_sorter = torch.sort(v_values, -1)

            u_weights = u_weights[..., u_sorter]
            v_weights = v_weights[..., v_sorter]

        u_cdf = torch.cumsum(u_weights, -1)
        v_cdf = torch.cumsum(v_weights, -1)

        cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)

        u_index = torch.searchsorted(u_cdf, cdf_axis)
        v_index = torch.searchsorted(v_cdf, cdf_axis)

        u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
        v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

        cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
        delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

        if p == 1:
            return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
        if p == 2:
            return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)  
        return torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)

    def g_circular(self, x, theta, radius=2):
        """
            https://github.com/kimiandj/gsw/blob/9f7f0ce6ae74049cb9ed753c34a6deff14cd4417/code/gsw/gsw.py#L149
        """
        theta = torch.stack([radius*th for th in theta])
        return torch.stack([torch.sqrt(torch.sum((x-th)**2,dim=1)) for th in theta],1)

    def get_powers(self, dim, degree):
        '''
        This function calculates the powers of a homogeneous polynomial
        e.g.
        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]
        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]

        https://github.com/kimiandj/gsw/blob/9f7f0ce6ae74049cb9ed753c34a6deff14cd4417/code/gsw/gsw.py#L149
        '''
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in self.get_powers(dim - 1,degree - value):
                    yield (value,) + permutation

    def homopoly(self, dim, degree):
        '''
        calculates the number of elements in a homogeneous polynomial

        https://github.com/kimiandj/gsw/blob/9f7f0ce6ae74049cb9ed753c34a6deff14cd4417/code/gsw/gsw.py#L149
        '''
        return len(list(self.get_powers(dim,degree)))

    def g_poly(self, X, theta, device, degree=3):
        ''' The polynomial defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
            degree: degree of the polynomial

            https://github.com/kimiandj/gsw/blob/9f7f0ce6ae74049cb9ed753c34a6deff14cd4417/code/gsw/gsw.py#L149
        '''
        N, d = X.shape
        assert theta.shape[1]==self.homopoly(d, degree)
        powers=list(self.get_powers(d, degree))
        HX=torch.ones((N, len(powers))).to(device)
        for k,power in enumerate(powers):
            for i,p in enumerate(power):
                HX[:,k]*=X[:,i]**p
        if len(theta.shape)==1:
            return torch.matmul(HX,theta)
        else:
            return torch.matmul(HX,theta.t())
    
    def sliced_cost(self, Xs, Xt, ftype="linear", projections=None,
                    u_weights=None, v_weights=None, p=1, degree=3):
        if projections is not None and ftype == "linear":
            Xps = (Xs @ projections).T
            Xpt = (Xt @ projections).T
        elif projections is not None and ftype == "circular":
            Xps = self.g_circular(Xs, projections.T).T
            Xpt = self.g_circular(Xt, projections.T).T
        elif projections is not None and ftype=="poly":
            Xps = self.g_poly(Xs, projections.T, device=Xs.device, degree=degree).T
            Xpt = self.g_poly(Xt, projections.T, device=Xt.device, degree=degree).T
        else:
            Xps = Xs.T
            Xpt = Xt.T

        return torch.mean(self.emd1D(Xps,Xpt,
                           u_weights=u_weights,
                           v_weights=v_weights,
                           p=p))
    
    def swd_loss(self, Xs, Xt, num_projections, device,
               u_weights=None, v_weights=None, p=1, 
               ftype="linear", degree=3): 
    
        num_features = Xs.shape[1]
        
        if ftype=="poly":
            dpoly = self.homopoly(num_features, degree)
            projections = np.random.normal(size=(dpoly, num_projections))
        else:
            projections = np.random.normal(size=(num_features, num_projections))
            
        projections = F.normalize(torch.from_numpy(projections), p=2, dim=0).type(Xs.dtype).to(device)

        return self.sliced_cost(Xs,Xt,projections=projections,
                           u_weights=u_weights,
                           v_weights=v_weights,
                           p=p, ftype=ftype, degree=degree)

####################################################################################################################################################################################
    def roll_by_gather(self, mat,dim, shifts: torch.LongTensor):
        ## https://stackoverflow.com/questions/66596699/how-to-shift-columns-or-rows-in-a-tensor-with-different-offsets-in-pytorch

        # assumes 2D array
        n_rows, n_cols = mat.shape

        if dim==0:
            arange1 = torch.arange(n_rows, device=mat.device).view((n_rows, 1)).repeat((1, n_cols))
            arange2 = (arange1 - shifts) % n_rows
            return torch.gather(mat, 0, arange2)
        elif dim==1:
            arange1 = torch.arange(n_cols, device=mat.device).view(( 1,n_cols)).repeat((n_rows,1))
            arange2 = (arange1 - shifts) % n_cols
            return torch.gather(mat, 1, arange2)

    def dCost(self, theta, u_values, v_values, u_cdf, v_cdf, p):
        v_values = v_values.clone()
        n = u_values.shape[-1]

        v_cdf_theta = v_cdf -(theta - torch.floor(theta))

        mask_p = v_cdf_theta>=0
        mask_n = v_cdf_theta<0

        v_values[mask_n] += torch.floor(theta)[mask_n]+1
        v_values[mask_p] += torch.floor(theta)[mask_p]

        if torch.any(mask_n) and torch.any(mask_p):
            v_cdf_theta[mask_n] += 1

        v_cdf_theta2 = v_cdf_theta.clone()
        v_cdf_theta2[mask_n] = np.inf
        shift = (-torch.argmin(v_cdf_theta2, axis=-1))

        v_cdf_theta = self.roll_by_gather(v_cdf_theta, 1, shift.view(-1,1))
        v_values = self.roll_by_gather(v_values, 1, shift.view(-1,1))
        v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)

        u_index = torch.searchsorted(u_cdf, v_cdf_theta)
        u_icdf_theta = torch.gather(u_values, -1, u_index.clip(0, n-1))

        ## Deal with 1
        u_cdfm = torch.cat([u_cdf, u_cdf[:,0].view(-1,1)+1], dim=1)
        u_valuesm = torch.cat([u_values, u_values[:,0].view(-1,1)+1],dim=1)
        u_indexm = torch.searchsorted(u_cdfm, v_cdf_theta, right=True)
        u_icdfm_theta = torch.gather(u_valuesm, -1, u_indexm.clip(0, n))

        dCp = torch.sum(torch.pow(torch.abs(u_icdf_theta-v_values[:,1:]), p)
                       -torch.pow(torch.abs(u_icdf_theta-v_values[:,:-1]), p), axis=-1)

        dCm = torch.sum(torch.pow(torch.abs(u_icdfm_theta-v_values[:,1:]), p)
                       -torch.pow(torch.abs(u_icdfm_theta-v_values[:,:-1]), p), axis=-1)

        return dCp.reshape(-1,1), dCm.reshape(-1,1)

    def Cost(self, theta, u_values, v_values, u_cdf, v_cdf, p):
        v_values = v_values.clone()

        m_batch, m = v_values.shape
        n_batch, n = u_values.shape

        v_cdf_theta = v_cdf -(theta - torch.floor(theta))

        mask_p = v_cdf_theta>=0
        mask_n = v_cdf_theta<0

        v_values[mask_n] += torch.floor(theta)[mask_n]+1
        v_values[mask_p] += torch.floor(theta)[mask_p]

        if torch.any(mask_n) and torch.any(mask_p):
            v_cdf_theta[mask_n] += 1

        ## Put negative values at the end
        v_cdf_theta2 = v_cdf_theta.clone()
        v_cdf_theta2[mask_n] = np.inf
        shift = (-torch.argmin(v_cdf_theta2, axis=-1))# .tolist()

        v_cdf_theta = self.roll_by_gather(v_cdf_theta, 1, shift.view(-1,1))
        v_values = self.roll_by_gather(v_values, 1, shift.view(-1,1))
        v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)  

        ## Compute abscisse
        cdf_axis, cdf_axis_sorter = torch.sort(torch.cat((u_cdf, v_cdf_theta), -1), -1)
        cdf_axis_pad = torch.nn.functional.pad(cdf_axis, (1, 0))
        delta = cdf_axis_pad[..., 1:] - cdf_axis_pad[..., :-1]

        ## Compute icdf
        u_index = torch.searchsorted(u_cdf, cdf_axis)
        u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))

        v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)
        v_index = torch.searchsorted(v_cdf_theta, cdf_axis)
        v_icdf = torch.gather(v_values, -1, v_index.clip(0, m))

        if p == 1:
            ot_cost = torch.sum(delta*torch.abs(u_icdf-v_icdf), axis=-1)
        elif p == 2:
            ot_cost = torch.sum(delta*torch.square(u_icdf-v_icdf), axis=-1)
        else:
            ot_cost = torch.sum(delta*torch.pow(torch.abs(u_icdf-v_icdf), p), axis=-1)
        return ot_cost

    def binary_search_circle(self, u_values, v_values, u_weights=None, v_weights=None, p=1, 
                             Lm=10, Lp=10, tm=-1, tp=1, eps=1e-6, require_sort=True):
        r"""
        Computes the Wasserstein distance on the circle using the Binary search algorithm proposed in [1].

        Parameters:
        u_values : ndarray, shape (n_batch, n_samples_u)
            samples in the source domain
        v_values : ndarray, shape (n_batch, n_samples_v)
            samples in the target domain
        u_weights : ndarray, shape (n_batch, n_samples_u), optional
            samples weights in the source domain
        v_weights : ndarray, shape (n_batch, n_samples_v), optional
            samples weights in the target domain
        p : float, optional
            Power p used for computing the Wasserstein distance
        Lm : int, optional
            Lower bound dC
        Lp : int, optional
            Upper bound dC
        tm: float, optional
            Lower bound theta
        tp: float, optional
            Upper bound theta
        eps: float, optional
            Stopping condition
        require_sort: bool, optional
            If True, sort the values.

        [1] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
        """
        ## Matlab Code : https://users.mccme.ru/ansobol/otarie/software.html

        n = u_values.shape[-1]
        m = v_values.shape[-1]

        device = u_values.device
        dtype = u_values.dtype

        if u_weights is None:
            u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

        if v_weights is None:
            v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

        if require_sort:
            u_values, u_sorter = torch.sort(u_values, -1)
            v_values, v_sorter = torch.sort(v_values, -1)

            u_weights = u_weights[..., u_sorter]
            v_weights = v_weights[..., v_sorter]

        u_cdf = torch.cumsum(u_weights, -1)
        v_cdf = torch.cumsum(v_weights, -1)

        L = max(Lm,Lp)

        tm = tm * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1,1)
        tm = tm.repeat(1, m)
        tp = tp * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1,1)
        tp = tp.repeat(1, m)
        tc = (tm+tp)/2

        done = torch.zeros((u_values.shape[0],m))

        cpt = 0
        while torch.any(1-done):
            cpt += 1

            dCp, dCm = self.dCost(tc, u_values, v_values, u_cdf, v_cdf, p)
            done = ((dCp*dCm)<=0) * 1

            mask = ((tp-tm)<eps/L) * (1-done)

            if torch.any(mask):
                ## can probably be improved by computing only relevant values
                dCptp, dCmtp = self.dCost(tp, u_values, v_values, u_cdf, v_cdf, p)
                dCptm, dCmtm = self.dCost(tm, u_values, v_values, u_cdf, v_cdf, p)
                Ctm = self.Cost(tm, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
                Ctp = self.Cost(tp, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)

                mask_end = mask * (torch.abs(dCptm-dCmtp)>0.001)
                tc[mask_end>0] = ((Ctp-Ctm+tm*dCptm-tp*dCmtp)/(dCptm-dCmtp))[mask_end>0]
                done[torch.prod(mask, dim=-1)>0] = 1
            ## if or elif?
            elif torch.any(1-done):
                tm[((1-mask)*(dCp<0))>0] = tc[((1-mask)*(dCp<0))>0]
                tp[((1-mask)*(dCp>=0))>0] = tc[((1-mask)*(dCp>=0))>0]
                tc[((1-mask)*(1-done))>0] = (tm[((1-mask)*(1-done))>0]+tp[((1-mask)*(1-done))>0])/2

        return self.Cost(tc.detach(), u_values, v_values, u_cdf, v_cdf, p)

    def emd1D_circle(self, u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
        n = u_values.shape[-1]
        m = v_values.shape[-1]

        device = u_values.device
        dtype = u_values.dtype

        if u_weights is None:
            u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

        if v_weights is None:
            v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

        if require_sort:
            u_values, u_sorter = torch.sort(u_values, -1)
            v_values, v_sorter = torch.sort(v_values, -1)

            u_weights = u_weights[..., u_sorter]
            v_weights = v_weights[..., v_sorter]

        if p == 1:
            ## Code inspired from https://gitlab.gwdg.de/shundri/circularOT/-/tree/master/
            values_sorted, values_sorter = torch.sort(torch.cat((u_values, v_values), -1), -1)

            cdf_diff = torch.cumsum(torch.gather(torch.cat((u_weights, -v_weights),-1),-1,values_sorter),-1)
            cdf_diff_sorted, cdf_diff_sorter = torch.sort(cdf_diff, axis=-1)

            values_sorted = torch.nn.functional.pad(values_sorted, (0,1), value=1)
            delta = values_sorted[..., 1:]-values_sorted[..., :-1]
            weight_sorted = torch.gather(delta, -1, cdf_diff_sorter)

            sum_weights = torch.cumsum(weight_sorted, axis=-1)-0.5
            sum_weights[sum_weights<0] = np.inf
            inds = torch.argmin(sum_weights, axis=-1)

            levMed = torch.gather(cdf_diff_sorted, -1, inds.view(-1,1))

            return torch.sum(delta * torch.abs(cdf_diff - levMed), axis=-1)

    def sp_sliced_cost(self, Xs, Xt, Us, p=2, u_weights=None, v_weights=None):
        """
            Parameters:
            Xs: ndarray, shape (n_samples_u, dim)
                Samples in the source domain
            Xt: ndarray, shape (n_samples_v, dim)
                Samples in the target domain
            Us: ndarray, shape (num_projections, d, 2)
                Independent samples of the Uniform distribution on V_{d,2}
            p: float
                Power
        """
        n_projs, d, k = Us.shape
        n, _ = Xs.shape
        m, _ = Xt.shape    


        ## Projection on S^1
        ## Projection on plane
        Xps = torch.matmul(torch.transpose(Us,1,2)[:,None], Xs[:,:,None]).reshape(n_projs, n, 2)
        Xpt = torch.matmul(torch.transpose(Us,1,2)[:,None], Xt[:,:,None]).reshape(n_projs, m, 2)

        ## Projection on sphere
        Xps = F.normalize(Xps, p=2, dim=-1)
        Xpt = F.normalize(Xpt, p=2, dim=-1)

        ## Get coords
        Xps = (torch.atan2(-Xps[:,:,1], -Xps[:,:,0])+np.pi)/(2*np.pi)
        Xpt = (torch.atan2(-Xpt[:,:,1], -Xpt[:,:,0])+np.pi)/(2*np.pi)

        if p==1:
            w1 = self.emd1D_circle(Xps, Xpt, u_weights=u_weights, v_weights=v_weights)
        else:
            w1 = self.binary_search_circle(Xps, Xpt, p=p, u_weights=u_weights, v_weights=v_weights)

        return torch.mean(w1)

    def sp_swd_loss(self, Xs, Xt, num_projections, device, u_weights=None, v_weights=None, p=2):
        """
            Compute the sliced-Wasserstein distance on the sphere.

            Parameters:
            Xs: ndarray, shape (n_samples_u, dim)
                Samples in the source domain
            Xt: ndarray, shape (n_samples_v, dim)
                Samples in the target domain
            num_projections: int
                Number of projections
            device: str
            p: float
                Power of SW. Need to be >= 1.
        """
        d = Xs.shape[1]

        ## Uniforms and independent samples on the Stiefel manifold V_{d,2}
        Z = torch.randn((num_projections,d,2), device=device)
        U, _ = torch.linalg.qr(Z)

        return self.sp_sliced_cost(Xs, Xt, U, p=p, u_weights=u_weights, v_weights=v_weights)

    def w2_unif_circle_approx(self, u_values):
        """
            Approximation 
            weights 1/n
            Compute u_values vs Uniform distribution

            Parameters:
            u_values: ndarray, shape (n_batch, n_samples)
        """
        n = u_values.shape[-1]

        u_values, _ = torch.sort(u_values, -1)
        u_weights = torch.full((n,), 1/n, dtype=u_values.dtype, device=u_values.device)
        u_cdf = torch.cumsum(u_weights, -1)

        alpha = torch.mean(u_values, axis=-1)-1/2

        ot_cost = torch.mean(torch.square(u_values-u_cdf-alpha[:,None]), axis=-1)
        return ot_cost

    def w2_unif_circle(self, u_values):
        """
            Closed-form

            weights 1/n
            Compute u_values vs Uniform distribution

            Parameters:
            u_values: ndarray, shape (n_batch, n_samples)
        """

        n = u_values.shape[-1]

        u_values, _ = torch.sort(u_values, -1)
        u_weights = torch.full((n,), 1/n, dtype=u_values.dtype, device=u_values.device)
        u_cdf = torch.cumsum(u_weights, -1)

        cpt1 = torch.mean(u_values**2, axis=-1)
        x_mean = torch.mean(u_values, axis=-1)

    #    ns = torch.tensor(range(1, n+1), dtype=torch.float)
    #    cpt2 = torch.sum((n+1-2*ns)*u_values, axis=-1)/n**2

        ns_n2 = torch.arange(n-1, -n, -2, dtype=torch.float, device=u_values.device)/n**2
        cpt2 = torch.sum(ns_n2 * u_values, dim=-1)

        return cpt1 - x_mean**2 +cpt2 + 1/12

    def sp_swd_unif_loss(self, Xs, num_projections, device):
        """
            Compute the SSW2 on the sphere w.r.t. a uniform distribution.

            Parameters:
            Xs: ndarray, shape (n_samples_u, dim)
                Samples in the source domain
            num_projections: int
                Number of projections
            device: str.
        """
        n, d = Xs.shape

        ## Uniforms and independent samples on the Stiefel manifold V_{d,2}
        Z = torch.randn((num_projections,d,2), device=device)
        U, _ = torch.linalg.qr(Z)

        ## Projection on S^1
        ## Projection on plane
        Xps = torch.matmul(torch.transpose(U,1,2)[:,None], Xs[:,:,None]).reshape(num_projections, n, 2)
        ## Projection on sphere
        Xps = F.normalize(Xps, p=2, dim=-1)
        ## Get coords
        Xps = (torch.atan2(-Xps[:,:,1], -Xps[:,:,0])+np.pi)/(2*np.pi)

        return torch.mean(self.w2_unif_circle(Xps))