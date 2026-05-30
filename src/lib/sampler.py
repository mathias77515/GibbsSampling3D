import numpy as np
import scipy
import time

class GibbsSampler:
    
    def __init__(self, H, w, d, wsqrt, maxiterCG):
        self.niterations = 1
        
        self.H     = H
        self.w     = w
        self.wsqrt = wsqrt
        self.d     = d
        self.TwiceNpix, self.TwiceNclouds = self.H.shape
        self.npix = int(self.TwiceNpix/2)
        self.nclouds = int(self.TwiceNclouds/2)
        self.A = H.T @ w @ H
        self.b = H.T @ w @ d
        
        self.maxiterCG = maxiterCG
        self.nsteps = 10
        
    def step(self, it):
        
        # --- Draw random vectors N(0, 1) ---
        omega      = np.random.normal(0, 1, size=2*self.npix)
        #omegaPrior = np.random.normal(0, 1, size=2*self.nclouds)
        
        # --- Draw for prior covariance ---
        #prior = np.sqrt(invCxDiag) * omegaPrior
        
        # --- Draw for data covariance ---
        rhs     = self.H.T @ (self.wsqrt @ omega)
        brandom = self.b + rhs #+ prior
        
        sol, _ = scipy.sparse.linalg.cg(self.A, brandom, x0=np.concatenate((self.AqSample[it-1], self.AuSample[it-1])), maxiter=self.maxiterCG)
        self.AqSample[it, ...] = sol[:self.nclouds]
        self.AuSample[it, ...] = sol[self.nclouds:2*self.nclouds]
            
    def initialize(self, niter):
        
        self.AqSample = np.zeros((niter, self.nclouds))
        self.AuSample = np.zeros((niter, self.nclouds))
        
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
        QSample   = QUSamples[:, :self.nclouds]
        USample   = QUSamples[:, self.nclouds:]
        return {
            "AqSample":self.AqSample, 
            "AuSample":self.AuSample,
            "pSample":np.sqrt(self.AqSample**2 + self.AuSample**2), 
            "psiSample":0.5 * np.arctan2(self.AuSample, self.AqSample),
            "QSample":QSample,
            "USample":USample
        }