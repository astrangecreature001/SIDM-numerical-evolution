"""Physical units and static baryon profiles, with G=c=M10=1 internally."""
from dataclasses import dataclass, asdict
import math
import numpy as np

G = 6.67430e-8
C_LIGHT = 2.99792458e10
MSUN = 1.988409870698051e33
PC = 3.0856775814913673e18
GYR = 3.15576e16
A_COLL = math.sqrt(16 / math.pi)
B_COLL = 25 * math.sqrt(math.pi) / 32
F10 = math.log(11) - 10 / 11
PSI_DM = (1 - math.log(11) / 10) / F10


def nfw_f(x):
    x = np.asarray(x, dtype=float)
    # Stable near the origin: ln(1+x)-x/(1+x).
    small = x < 1e-3
    out = np.log1p(x) - x / (1 + x)
    series = sum((-1.)**k * (k - 1) / k * x**k for k in range(2, 9))
    return np.where(small, series, out)


def bisect(function, lower, upper, iterations=80):
    lo,hi=float(lower),float(upper);flo=function(lo)
    if flo*function(hi)>0: raise ValueError('Root is not bracketed')
    for _ in range(iterations):
        mid=.5*(lo+hi)
        if flo*function(mid)<=0:hi=mid
        else:lo=mid;flo=function(lo)
    return .5*(lo+hi)


def nfw_inverse(mass_fraction):
    target=np.asarray(mass_fraction)*F10
    lo=np.zeros_like(target);hi=np.full_like(target,10.)
    for _ in range(64):
        mid=.5*(lo+hi);left=nfw_f(mid)<target
        lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
    return .5*(lo+hi)


@dataclass(frozen=True)
class Units:
    M10_Msun: float
    rs_kpc: float
    sigma_cm2_g: float = 5.

    @property
    def length_cm(self): return G * self.M10_Msun * MSUN / C_LIGHT**2
    @property
    def time_s(self): return self.length_cm / C_LIGHT
    @property
    def R0(self): return self.rs_kpc * 1000 * PC / self.length_cm
    @property
    def sigma(self): return self.sigma_cm2_g * self.M10_Msun * MSUN / self.length_cm**2
    @property
    def rho_s_cgs(self): return self.M10_Msun * MSUN / (4 * math.pi * (self.rs_kpc * 1000 * PC)**3 * F10)
    @property
    def rho_s(self): return self.rho_s_cgs * self.length_cm**3 / (self.M10_Msun * MSUN)
    @property
    def v_s_cgs(self): return self.rs_kpc * 1000 * PC * math.sqrt(4 * math.pi * G * self.rho_s_cgs)
    @property
    def t0_s(self): return 1 / (A_COLL * self.sigma_cm2_g * self.rho_s_cgs * self.v_s_cgs)
    @property
    def t0(self): return self.t0_s / self.time_s

    def metadata(self):
        return dict(**asdict(self), G_cgs=G, c_cgs=C_LIGHT, Msun_g=MSUN, pc_cm=PC,
                    Gyr_s=GYR, a=A_COLL, b=B_COLL, R0=self.R0, sigma_geometric=self.sigma,
                    time_unit_s=self.time_s, rho_s_cgs=self.rho_s_cgs,
                    t0_Gyr=self.t0_s / GYR, t0_geometric=self.t0,
                    t0_definition='1 / (sqrt(16/pi) * sigma_m * rho_s * rs * sqrt(4*pi*G*rho_s))')


@dataclass(frozen=True)
class Baryons:
    profile: str = 'none'
    mu: float = 0.
    eta: float = 1.

    def mass(self, x):
        x = np.asarray(x, dtype=float)
        if self.profile == 'none': return np.zeros_like(x)
        if self.mu < 0 or self.eta <= 0: raise ValueError('Invalid static baryon parameters')
        if self.profile == 'feng2021-softened': return self.mu * x**3 / (x*x + self.eta**2)**1.2
        if self.profile == 'plummer': return self.mu * x**3 / (x*x + self.eta**2)**1.5
        if self.profile == 'hernquist': return self.mu * x*x / (x + self.eta)**2
        raise ValueError(self.profile)

    def depth(self):
        if self.profile == 'none': return 0.
        e = self.eta
        if self.profile == 'feng2021-softened': v = (e**(-.4) - (100 + e*e)**(-.2)) / .4
        elif self.profile == 'plummer': v = 1/e - 1/math.sqrt(100 + e*e)
        elif self.profile == 'hernquist': v = 1/e - 1/(10 + e)
        else: raise ValueError(self.profile)
        return self.mu * v / PSI_DM


def from_config(config):
    h = config['halo']
    u = Units(h['M10_Msun'], h['rs_kpc'], config['sigma_cm2_g'])
    b = Baryons(config['profile'], config['mu_b'], config['eta_b'] or 1.)
    return u, b
