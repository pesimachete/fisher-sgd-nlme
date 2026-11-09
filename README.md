# pesimachete/fisher-sgd-nlme

Welcome to the **pesimachete/fisher-sgd-nlme** repository. This project focuses on parameter estimation for Nonlinear Mixed Effects (NLME) models using Fisher scoring, stochastic gradient descent (SGD), and normalizing flows.

## 1. Choosing a Model (`model.py`)

Inside the `model.py` file, there are 3 distinct models provided for generating the latent individual parameters:

1. Coordinate-wise three-component Gaussian mixtures

2. Single-moon construction

3. Empty-ball construction (a uniform spherical shell)

Before running the code, you must choose which one to use. To do this, open `model.py` and **uncomment** the specific model you want to run, while ensuring the other two are commented out.

## 2. Training Pipeline

### Step 1: Fisher SGD Optimization

First, you run the main optimization using the Fisher SGD algorithms to estimate the foundational population and individual parameters.

### Step 2: Flow Map Learning

After the Fisher SGD step, if you want, you can make the normalizing flow learn by the map. This map learning phase is executed using:

* **`flow_training.py`**: Trains the normalizing flow to learn the map of the complex latent geometries.

## 3. Flow Plots and Visualization

Once the learning phases are complete, you can run the flow plots to visualize the results. The evaluation and plotting phase is handled using **`flow_evaluation.py`** alongside the unified plotting script. The functions in these files serve to:

* **Evaluate the Flow Map:** (`flow_evaluation.py`) Evaluates the performance and accuracy of the learned flow map.
* **Visualize Latent Densities:** Plot the distribution of individual parameters to see how accurately the flow captures complex geometries (like the Gaussian mixture, moon, or spherical shell).
* **Trajectory Fitting:** Compare the predicted logistic growth curves against the actual observed data points for the individuals.
* **Diagnostics:** Show the loss curves, convergence metrics, and parameter update trajectories over the training iterations.

## References

For detailed mathematical formulations, likelihood definitions, and sampling constructions, please refer to the provided documentation:

* *Generation of Individual Parameters for the Logistic Growth Model*

* *Mathematical Formalization of the Logistic Growth Nonlinear Mixed Effects Model*
