import functools

import jax
import jax.numpy as jnp

import parametrization_cookbook.jax as pc
from parametrization_cookbook.functions.jax import expit

parametrization = pc.NamedTuple(
    asymptotic=pc.Real(scale=100),
    inflexion=pc.Real(loc=100, scale=100),
    tau=pc.RealPositive(scale=100),
    # cov_latent=pc.MatrixDiagPosDef(dim=2, scale=(100, 100)),
    cov_latent=pc.MatrixSymPosDef(dim=3, scale=(100, 100, 100)),
    var_residual=pc.RealPositive(scale=100),
)


@jax.jit
def log_likelihood_rows(theta, z, y, t):
    p = parametrization.reals1d_to_params(theta)
    n, J = y.shape
    assert z.shape == (n, 3)
    assert t.shape == (J,)

    mean = jnp.array((p.asymptotic, p.inflexion, p.tau))
    dz = z - mean
    log_likli_latent = (
        -0.5 * jnp.linalg.slogdet(p.cov_latent)[1]
        - 0.5 * 3 * jnp.log(2 * jnp.pi)
        - 0.5 * ((dz @ jnp.linalg.inv(p.cov_latent)) * dz).sum(axis=1)
    )

    dy = y - z[:, 0][:, None] * expit((t[None, :] - z[:, 1][:, None]) / z[:, 2][:, None])
    log_likli_obs = (
        -0.5 * jnp.log(2 * jnp.pi * p.var_residual) * J
        - 0.5 * (dy**2).sum(axis=1) / p.var_residual
    )

    return log_likli_latent + log_likli_obs


jac_log_likelihood_rows = jax.jit(jax.jacfwd(log_likelihood_rows))


@jax.jit
def log_likelihood(theta, z, y, t):
    return log_likelihood_rows(theta, z, y, t).sum()

grad_log_likelihood = jax.jit(jax.grad(log_likelihood))

@jax.jit
def mh_step(theta, z, y, t, sigma_proposal, prng_key):
    p = parametrization.reals1d_to_params(theta)
    n, J = y.shape
    assert z.shape == (n, 3)
    assert t.shape == (J,)
    assert sigma_proposal.shape == (3,)

    key1, key2 = jax.random.split(prng_key)
    z_propo = z + jax.random.normal(key1, shape=(n, 3)) * sigma_proposal
    log_likli = log_likelihood_rows(theta, z, y, t)
    log_likli_propo = log_likelihood_rows(theta, z_propo, y, t)

    mask = log_likli_propo - log_likli > jnp.log(
        jax.random.uniform(key=key2, shape=(n,))
    )
    ret_z = z_propo * mask[:, None] + z * (1 - mask[:, None])
    return mask.sum(), ret_z

 
@jax.jit
def mh_step_gibbs(theta, z, y, t, sigma_proposal, prng_key):
    p = parametrization.reals1d_to_params(theta)
    n, J = y.shape
    assert z.shape == (n, 3)
    assert t.shape == (J,)
    assert sigma_proposal.shape == (3,)

    log_likli = log_likelihood_rows(theta, z, y, t)
    key1, key2, key3, key4, key5, key6  = jax.random.split(prng_key, 6)

    z_propo = z.at[:, 0].add(jax.random.normal(key1, shape=(n,)) * sigma_proposal[0])
    log_likli_propo = log_likelihood_rows(theta, z_propo, y, t)

    mask = log_likli_propo - log_likli > jnp.log(
        jax.random.uniform(key=key2, shape=(n,))
    )
    assert mask.shape == (n,)
    assert z_propo.shape == z.shape
    z = z_propo * mask[:, None] + z * (1 - mask[:, None])
    log_likli = log_likli_propo * mask + log_likli * (1 - mask)
    ar1 = mask.sum()

    z_propo = z.at[:, 1].add(jax.random.normal(key3, shape=(n,)) * sigma_proposal[1])
    log_likli_propo = log_likelihood_rows(theta, z_propo, y, t)

    mask = log_likli_propo - log_likli > jnp.log(
        jax.random.uniform(key=key4, shape=(n,))
    )
    z = z_propo * mask[:, None] + z * (1 - mask[:, None])
    ar2 = mask.sum()

    z_propo = z.at[:, 2].add(jax.random.normal(key5, shape=(n,)) * sigma_proposal[2])
    log_likli_propo = log_likelihood_rows(theta, z_propo, y, t)

    mask = log_likli_propo - log_likli > jnp.log(
        jax.random.uniform(key=key6, shape=(n,))
    )
    z = z_propo * mask[:, None] + z * (1 - mask[:, None])
    log_likli = log_likli_propo * mask + log_likli * (1 - mask)
    ar3 = mask.sum()

    return jnp.array([ar1, ar2, ar3]), z


@functools.partial(jax.jit, static_argnums=(1,))
def simu_data(theta, n, t=20, prng_key=0):
    if prng_key.ndim == 0:
        prng_key = jax.random.PRNGKey(prng_key)

    if isinstance(t, int):
        t = jnp.linspace(100, 1500, t)
    (J,) = t.shape

    p = parametrization.reals1d_to_params(theta)
    key1, key2 = jax.random.split(prng_key)
    z = jax.random.normal(key=key1, shape=(n, 3)) @ jnp.linalg.cholesky(
        p.cov_latent
    ).T + jnp.array((p.asymptotic, p.inflexion, p.tau))
    y = z[:, 0][:, None] * expit(
        (t[None, :] - z[:, 1][:, None]) /z[:, 2][:, None]
    ) + jax.random.normal(key=key2, shape=(n, J)) * jnp.sqrt(p.var_residual)
    return z, y, t
