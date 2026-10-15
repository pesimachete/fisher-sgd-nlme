import algos
import model
import jax.numpy as jnp
import jax
import pickle

import pandas as pd

from tqdm import tqdm
from collections import namedtuple

def sample_and_estim(theta0, n, prng_key):
    key_simu, key_estim = jax.random.split(prng_key)
    z, y, t = model.simu_data(theta0, n, prng_key=key_simu)
    y_pd = pd.DataFrame(y,columns=["x_" + str(i) for i in range(1,21)])
    y_pd.to_csv('data/y'+str(key_simu)+'.txt')
    res = algos.estim(y, t, stop_crit=1e-6, N_smooth=5000, prng_key=key_estim, smart_start=2000)
    return res


theta0 = model.parametrization.params_to_reals1d(
    asymptotic=200,
    inflexion=500,
    tau=150,
    cov_latent=jnp.diag(jnp.array([40, 100, 100])),
    var_residual=100,
)

covariates = namedtuple("covariates","t")
cov = covariates(jnp.linspace(100,1500,20))

Nsimus = 1000
n = 1000
keyy = 0
many_res = list(
    tqdm(
        (
            sample_and_estim(theta0, n, key)
            for key in jax.random.split(jax.random.PRNGKey(keyy), Nsimus)
        ),
        total=Nsimus,
        smoothing=1000,
    )
)

with open('example_res.npy', 'wb') as f:
    pickle.dump(many_res[0],f)


theta = jnp.array([t for t in ((x.theta) for x in many_res)])
fim = jnp.array([f for f in (x.fisher_info_mat for x in many_res)])
    
    
with open('theta_all_%s.npy' % keyy, 'wb') as f:
    jnp.save(f, theta)
    
with open('fim_all_%s.npy' % keyy, 'wb') as f:
    jnp.save(f, fim)

