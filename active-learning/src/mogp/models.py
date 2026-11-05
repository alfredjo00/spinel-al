"""GP surrogates for the spinel landscape: ICM, batch-independent, and scalar.

`ICMGPRegressor` is the new model; the other two are the baselines it has to
beat. All three keep the same `__init__(train_x, train_y, ...)` / `.predict(x)`
shape as `al_functions.GPRegressor` so they can drop into
`active_learning_HEO.py` once the benchmark settles the design.
"""
import gpytorch
import torch
from gpytorch import kernels


class ICMGPRegressor(gpytorch.models.ExactGP):
    """Intrinsic coregionalization model over the Gibbs landscape.

    K = k_Matern-5/2,ARD(x, x') ⊗ B,  with  B = W W^T + diag(kappa),  W in R^{T x rank}.

    Because every composition is evaluated at every vacancy level, the design is
    a complete grid and GPyTorch keeps the Kronecker structure: inference costs
    O(N^3 + T^3) rather than O((N*T)^3).

    Notes
    -----
    - `MultitaskMean` gives each task its own constant. This is not optional:
      the raw task means span 0 to 10.5 eV, and a single shared constant would
      push all of that offset into the kernel.
    - The likelihood uses `rank=0`, i.e. an independent noise variance per task
      plus one global term. That preserves the Kronecker fast path and lets the
      model learn the noise ramp with vacancy level (0.2 -> 0.8 eV in
      `al_config.build_ensemble_noise_std`) instead of having it hard-coded.
    - `rank` is the coregionalization rank, not the likelihood rank. PCA on the
      740 training landscapes puts 97.7% of the variance in PC1, but OTC depends
      on where dG/dv crosses zero, i.e. on the curvature residual in PC2/PC3 --
      so pick `rank` on downstream OTC error, never on Gibbs RMSE.
    """

    def __init__(self, train_x, train_y, num_tasks=None, rank=3, likelihood=None,
                 device=None, max_cholesky=4096):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if num_tasks is None:
            num_tasks = train_y.size(-1)

        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=num_tasks, rank=0, has_global_noise=True
            )

        train_x, train_y = train_x.to(device), train_y.to(device)
        super().__init__(train_x, train_y, likelihood.to(device))

        self.device = device
        self.num_tasks = num_tasks
        self.rank = rank
        self.likelihood = likelihood
        # Carried on the model so no call site can forget it. GPyTorch defaults
        # to 800 and silently falls back to CG above that; CG does not converge
        # on this problem (residual ~1e5 against a tolerance of 1) and is also
        # ~100x slower than the exact Cholesky path.
        self.max_cholesky = max_cholesky

        input_dim = train_x.size(-1)
        self.mean = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        ).to(device)
        self.kernel = kernels.MultitaskKernel(
            kernels.ScaleKernel(kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)),
            num_tasks=num_tasks,
            rank=rank,
        ).to(device)

    def forward(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(self.mean(x), self.kernel(x))

    def predict_decomposed(self, x, chunk_size=256):
        """Posterior mean plus the epistemic/total std split, in one pass.

        Returns (mean [N, T], epistemic_std [N, T], total_std [N, T]) where
        epistemic is the std of the latent f and total is the std of y, i.e.
        with the learned per-task and global observation noise added. For a GP
        this split is exact, unlike the member-spread / mean-predicted-variance
        estimate a deep ensemble reports.

        Deliberately does NOT use `gpytorch.settings.fast_pred_var()`. LOVE
        builds a *training-sized* root cache and densifies it: at N=1824 train
        and T=14 that is a 25536^2 matrix, ~2.6 GB, which OOMs a 40 GB A100
        even though training itself exploits the Kronecker structure fine. The
        standard prediction path solves against the test-train covariance
        instead and keeps the structure.
        """
        self.eval()
        means, epistemic, total = [], [], []
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky):
            for start in range(0, len(x), chunk_size):
                latent = self(x[start:start + chunk_size].to(self.device))
                observed = self.likelihood(latent)
                means.append(observed.mean.cpu())
                epistemic.append(latent.stddev.cpu())
                total.append(observed.stddev.cpu())
        return torch.cat(means), torch.cat(epistemic), torch.cat(total)

    def predict(self, x, chunk_size=256):
        """Posterior predictive p(y*|x*) as (mean [N, T], std [N, T])."""
        mean, _, total = self.predict_decomposed(x, chunk_size)
        return mean, total

    def sample(self, x, n_samples=1, chunk_size=256, observation_noise=False):
        """Joint draws over tasks, shape [n_samples, N, T].

        Each draw is a coherent landscape -- the whole point of the ICM here.
        The quadratic fit downstream reads curvature in delta, so it needs a
        smooth G(v) curve, not T independently jittered points.

        Parameters
        ----------
        observation_noise : bool, default False
            False draws the latent f (epistemic only). This is what Thompson
            sampling wants, and it matches the deep ensemble, whose exploration
            comes from member disagreement rather than predicted aleatoric
            variance. True adds the learned per-task noise, which is applied
            *independently per task* and therefore destroys the smoothness in
            delta that the quadratic fit depends on -- with per-task noise in
            the eV range the fitted minimum jitters and Delta v becomes mostly
            noise. Only set True to reproduce the scalar GP's convention of
            drawing from p(y*|.) including sigma_n^2.

        Notes
        -----
        Draws are joint across tasks but independent across candidates, matching
        how `samples_GP` already draws for the scalar GP baseline. A fully
        coherent function draw (joint across candidates too, as the DE gets by
        picking one member) would need pathwise conditioning; the ICM makes that
        cheap -- rank latent draws plus T residuals -- but it changes the
        acquisition semantics, so it belongs in an ablation, not the main run.
        """
        self.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), chunk_size):
                pred = self(x[start:start + chunk_size].to(self.device))
                if observation_noise:
                    pred = self.likelihood(pred)
                out.append(pred.rsample(torch.Size([n_samples])).cpu())
        return torch.cat(out, dim=1)

    def task_noise_std(self):
        """Learned per-task observation std, in normalised units, shape [T].

        Returns None if the installed GPyTorch names the per-task noise
        differently; this is diagnostic output only, never worth losing a job.
        """
        noises = getattr(self.likelihood, 'task_noises', None)
        if noises is None:
            return None
        return noises.detach().cpu().sqrt().squeeze()

    def task_correlation(self):
        """Correlation implied by B = W W^T + diag(kappa), shape [T, T]."""
        try:
            B = self.kernel.task_covar_module.covar_matrix.to_dense()
        except AttributeError:
            return None
        B = B.detach().cpu().squeeze()
        d = B.diagonal().clamp_min(1e-12).sqrt()
        return B / torch.outer(d, d)


class IndependentMOGPRegressor(gpytorch.models.ExactGP):
    """T independent GPs sharing one training loop (batch-independent multitask).

    The ablation that isolates what coregionalization actually buys: same
    multi-output target representation as the ICM, no cross-task correlation.
    Equivalent to the `use_gp` + `predict_landscape` path already in
    `active_learning_HEO.py:200`, but batched so it trains in one pass.
    """

    def __init__(self, train_x, train_y, num_tasks=None, likelihood=None, device=None,
                 max_cholesky=4096):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if num_tasks is None:
            num_tasks = train_y.size(-1)

        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

        train_x, train_y = train_x.to(device), train_y.to(device)
        super().__init__(train_x, train_y, likelihood.to(device))

        self.device = device
        self.num_tasks = num_tasks
        self.likelihood = likelihood
        # Shared with the inherited predict methods below; see ICMGPRegressor.
        self.max_cholesky = max_cholesky

        input_dim = train_x.size(-1)
        batch_shape = torch.Size([num_tasks])
        self.mean = gpytorch.means.ConstantMean(batch_shape=batch_shape).to(device)
        self.kernel = kernels.ScaleKernel(
            kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim, batch_shape=batch_shape),
            batch_shape=batch_shape,
        ).to(device)

    def forward(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(self.mean(x), self.kernel(x))
        )

    predict_decomposed = ICMGPRegressor.predict_decomposed
    predict = ICMGPRegressor.predict
    sample = ICMGPRegressor.sample


class ScalarGPRegressor(gpytorch.models.ExactGP):
    """The current paper baseline: single GP on scalar OTC (al_functions.py:266)."""

    def __init__(self, train_x, train_y, likelihood=None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        train_x, train_y = train_x.to(device), train_y.to(device)
        super().__init__(train_x, train_y, likelihood.to(device))

        self.device = device
        self.likelihood = likelihood
        self.mean = gpytorch.means.ConstantMean().to(device)
        self.kernel = kernels.ScaleKernel(
            kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1))
        ).to(device)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.kernel(x))

    def predict(self, x, chunk_size=512):
        self.eval()
        means, stds = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start in range(0, len(x), chunk_size):
                pred = self.likelihood(self(x[start:start + chunk_size].to(self.device)))
                means.append(pred.mean.cpu())
                stds.append(pred.stddev.cpu())
        return torch.cat(means), torch.cat(stds)


def fit_gp(model, train_x, train_y, n_iters=400, lr=0.01, patience=None, verbose=False, logger=None):
    """Maximise the exact marginal log likelihood with Adam.

    Fewer iterations than the scalar GP's 1250 (al_config) because the multitask
    MLL is a lot more expensive per step; `patience` stops early once the loss
    plateaus. Returns the loss trace.
    """
    model.train()
    model.likelihood.train()

    train_x = train_x.to(model.device)
    train_y = train_y.to(model.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    losses, best, since_best = [], float('inf'), 0
    for i in range(n_iters):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if loss.item() < best - 1e-4:
            best, since_best = loss.item(), 0
        else:
            since_best += 1

        if (verbose or logger) and (i % 50 == 0 or i == n_iters - 1):
            msg = f'iter {i:4d}  loss {loss.item():.4f}'
            logger.info(msg) if logger else print(msg)

        if patience is not None and since_best >= patience:
            if logger:
                logger.info(f'early stop at iter {i} (loss {loss.item():.4f})')
            break

    return losses
