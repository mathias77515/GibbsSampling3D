import numpy as np
import scipy
import time

class GibbsSampler:
    
    def __init__(self, H, w, d, wsqrt, maxiterCG, sigmaPrior=0.1, muPrior=None):
        self.niterations = 1
        
        
        self.H     = H
        self.w     = w
        self.wsqrt = wsqrt
        self.d     = d
        self.TwiceNpix, self.TwiceNclouds = self.H.shape
        self.npix = int(self.TwiceNpix/2)
        self.nclouds = int(self.TwiceNclouds/2)

        # muPrior: array of shape (2*nclouds,) or None
        if muPrior is None:
            self.muPrior = np.zeros(2*self.nclouds)
        else:
            self.muPrior = muPrior
        
        self.Cx = np.eye(2*self.nclouds) * sigmaPrior**2
        self.invCxDiag = 1 / np.diag(self.Cx)

        self.A = (H.T @ w @ H) + scipy.sparse.diags(self.invCxDiag)
        print(f"H^T W H diag mean : {np.nanmean((H.T @ w @ H).diagonal()):.4e}")
        print(f"invCx mean        : {np.nanmean(self.invCxDiag):.4e}")
        print(f"ratio mean        : {np.nanmean((H.T @ w @ H).diagonal()/self.invCxDiag):.4e}")

        self.b = H.T @ w @ d + self.invCxDiag * self.muPrior
        
        self.maxiterCG = maxiterCG
        self.nsteps = 10
        
    def step(self, it):
        
        # --- Draw random vectors N(0, 1) ---
        omega      = np.random.normal(0, 1, size=2*self.npix)
        omegaPrior = np.random.normal(0, 1, size=2*self.nclouds)
        
        # --- Draw for prior covariance ---
        prior = np.sqrt(self.invCxDiag) * omegaPrior
        
        # --- Draw for data covariance ---
        rhs     = self.H.T @ (self.wsqrt @ omega)
        brandom = self.b + rhs + prior
        
        sol, _ = scipy.sparse.linalg.cg(self.A, brandom, x0=np.concatenate((self.AqSample[it-1], self.AuSample[it-1])), maxiter=self.maxiterCG)
        self.AqSample[it, ...] = sol[:self.nclouds]
        self.AuSample[it, ...] = sol[self.nclouds:2*self.nclouds]
            
    def initialize(self, niter):
        
        self.AqSample = np.zeros((niter, self.nclouds))
        self.AuSample = np.zeros((niter, self.nclouds))
        psi0 = np.random.uniform(-1, 1, size=self.nclouds)
        self.AqSample[0] = 0.1 * np.cos(2 * psi0)
        self.AuSample[0] = 0.1 * np.sin(2 * psi0)
         
    def convergence(self, it, niter):
        
        # --- Periodic diagnostics ---
        if it % self.nsteps == 0:
            t_now = time.time() - self.timeGibbs
            eta   = t_now / it * (niter - it)

            QUsim = self.H @ np.concatenate((self.AqSample[it], self.AuSample[it]))
            Qsim  = QUsim[:self.nclouds]
            Usim  = QUsim[self.nclouds:]
            dsim  = np.concatenate((Qsim, Usim))
            chi2  = np.nansum(self.w * (self.d - dsim)**2)
            chi2R = chi2 / (2 * self.npix)
            
            print(
                f"[{it:03d}/{niter:03d}] "
                f"χ²_red: {chi2R:8.2f}  "
                f"Elapsed: {t_now:8.2f}s  "
                f"ETA: {eta:8.2f}s"
            )
            
    def run(self, niter):
        
        # --- Initiate the state ---
        self.initialize(niter)
        
        self.timeGibbs = time.time()
        for it in range(1, niter):
            
            # --- One Gibbs step ---
            self.step(it)
            
            # --- Convergence diagnosis ---
            self.convergence(it, niter)

        QUSamples = np.array([self.H @ np.concatenate((self.AqSample[it], self.AuSample[it])) for it in range(niter)])
        
        QSample   = QUSamples[:, :self.npix]
        USample   = QUSamples[:, self.npix:]

        return {
            "AqSample":self.AqSample, 
            "AuSample":self.AuSample,
            "pSample":np.sqrt(self.AqSample**2 + self.AuSample**2), 
            "psiSample":0.5 * np.arctan2(self.AuSample, self.AqSample),
            "QSample":QSample,
            "USample":USample
        }