import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from tqdm import trange

from flowjax.distributions import StandardNormal, Transformed
from flowjax.bijections import Coupling, RationalQuadraticSpline

# ==========================================
# 1. Define the 3D Fixed Dataset
# ==========================================
def sample_3d_data(key, num_samples):
    k1, k2, k3, k4, noise_key = jr.split(key, 5)
    
    # Dim 0 and 1: Standard Normal (Conditioning variables)
    x0 = jr.normal(k1, (num_samples,))
    x1 = jr.normal(k2, (num_samples,))
    
    # Dim 2: The Unbalanced GMM
    means = jnp.array([-3.0, 0.0, 3.0])
    stds = jnp.array([0.5, 0.4, 0.5])
    weights = jnp.array([0.25,0.5,0.25]) # Only the center peak
    
    components = jr.choice(k3, 3, shape=(num_samples,), p=weights)
    normals = jr.normal(k4, shape=(num_samples,))
    x2_clean = means[components] + stds[components] * normals
    
    # Add noise to Dim 2 to induce error
    x2 = x2_clean + 0.5 * jr.normal(noise_key, shape=(num_samples,))
    
    # Stack into 3D dataset (Shape: [num_samples, 3])
    return jnp.stack([x0, x1, x2], axis=-1)

# Generate the fixed training set
key = jr.PRNGKey(42)
key, subkey = jr.split(key)
num_samples = 20000
x_train = sample_3d_data(subkey, num_samples)

print(f"Generated {num_samples} 3D samples.")
print(f"Shape: {x_train.shape}")
print(f"First sample [x0, x1, x2]: {x_train[0]}\n")

# ==========================================
# 2. Define the 3D Conditional Normalizing Flow
# ==========================================
bound = 5.0
base_dist = StandardNormal((3,))
K = 3

# Coupling layer automatically handles the neural network internally
bijection = Coupling(
    key=jr.PRNGKey(127),
    transformer=RationalQuadraticSpline(knots=K, interval=bound),
    untransformed_dim=2,  # Dims 0 and 1 are passed through unchanged
    dim=3,                # Total dimensions
    nn_width=64,          # Width of the internal neural network
    nn_depth=3            # Depth of the internal neural network
)

flow = Transformed(base_dist, bijection)

# ==========================================
# 3. Define Adam Optimization
# ==========================================
@eqx.filter_value_and_grad
def loss_fn(model, x):
    log_probs = jax.vmap(model.log_prob)(x)
    return -jnp.mean(log_probs)

@eqx.filter_jit
def train_step(model, state, x):
    loss, grads = loss_fn(model, x)
    updates, new_state = optim.update(grads, state)
    model = eqx.apply_updates(model, updates)
    return model, new_state, loss

learning_rate = 0.01  
optim = optax.adam(learning_rate)
opt_state = optim.init(eqx.filter(flow, eqx.is_inexact_array))

# ==========================================
# 4. Mini-Batch Training Loop (Adam with tqdm)
# ==========================================
epochs = 1000
batch_size = 1024
num_batches = num_samples // batch_size

print("Training starting using Adam...")
# Initialize tqdm progress bar
pbar = trange(epochs, desc="Training Flow")

for epoch in pbar:
    key, subkey = jr.split(key)
    indices = jr.permutation(subkey, num_samples)
    
    epoch_loss = 0.0
    for i in range(num_batches):
        batch_indices = indices[i * batch_size : (i + 1) * batch_size]
        x_batch = x_train[batch_indices]
        
        flow, opt_state, loss = train_step(flow, opt_state, x_batch)
        epoch_loss += loss
        
    avg_loss = epoch_loss / num_batches
    
    # Update the progress bar with the current average loss
    pbar.set_postfix(loss=f"{avg_loss:.4f}")

# ==========================================
# 5. Conditional Visualization
# ==========================================
x2_eval = jnp.linspace(-bound, bound, 1000)

# Create 3D test points: [0.0, 0.0, x2]
x_test = jnp.stack([
    jnp.zeros_like(x2_eval), 
    jnp.zeros_like(x2_eval), 
    x2_eval
], axis=-1)

# 1. Obter a probabilidade conjunta (log) para os pontos 3D
joint_log_prob = jax.vmap(flow.log_prob)(x_test)

# 2. Calcular a probabilidade marginal (log) das condições X0=0 e X1=0
# Como usamos StandardNormal, a probabilidade é apenas o logpdf no ponto 0.0
marginal_cond_log_prob = jsp.stats.norm.logpdf(0.0) * 2

# 3. Subtrair para obter a probabilidade condicional: P(X2 | X0, X1) = P(X0, X1, X2) / P(X0, X1)
learned_log_prob = joint_log_prob - marginal_cond_log_prob
learned_pdf = jnp.exp(learned_log_prob)

# --- Plotting ---
plt.figure(figsize=(10, 6))

plt.hist(x_train[:, 2], bins=100, density=True, alpha=0.5, color='gray', label="Training Data (Dim 2)")
plt.plot(x2_eval, learned_pdf, 'r-', linewidth=2.5, label="Learned Conditional PDF $P(X_2 | X_0=0, X_1=0)$")

plt.title("3D Conditional Spline Flow (Dimension 2)")
plt.xlabel("Target Space (x2)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
