# SDA Python Implementation Patterns

## Overview

This skill covers Python implementation patterns for Sequential Decision Analytics (SDA) using Warren B. Powell's framework. It provides code patterns, architecture guidance, and working examples for building and evaluating sequential decision models.

All patterns are based on:
- The `SDPModel` and `SDPPolicy` abstract base classes from the `djanka2/stochastic-optimization` repository
- Code patterns from Powell's original library at `wbpowell328/stochastic-optimization`
- Community notebooks at `Peymankor/seqdec_powell_repo`

The goal is to provide a consistent, composable architecture for modeling sequential decision problems under uncertainty, evaluating policies via simulation, and comparing the four policy classes (PFA, CFA, VFA, DLA).

---

## Core Architecture: SDPModel and SDPPolicy

Every SDA implementation follows a two-class architecture: a **model** that defines the problem and a **policy** that defines the decision rule.

### The SDPModel Abstract Base Class

`SDPModel` encapsulates the five core components of any sequential decision problem: state variables, decision variables, exogenous information, the transition function, and the objective function.

```python
from abc import ABC, abstractmethod

class SDPModel(ABC):
    """Base class for Sequential Decision Problems.

    Every sequential decision problem is defined by five components:
      1. State variables S_t
      2. Decision variables x_t
      3. Exogenous information W_{t+1}
      4. Transition function S_{t+1} = S^M(S_t, x_t, W_{t+1})
      5. Objective function (contribution/cost) C(S_t, x_t)

    Subclass this to implement a specific problem.
    """

    @abstractmethod
    def build_state(self, info: dict) -> dict:
        """Define state variables S_t.

        Returns a dict with:
        - 'physical': Physical/resource state R_t (e.g., inventory level,
          asset position, resource allocations)
        - 'belief': Belief state B_t (if applicable; e.g., posterior
          distribution parameters for unknown quantities)
        - 'other': Other information (e.g., time-of-day, exogenous
          context variables)
        """
        pass

    @abstractmethod
    def exogenous_info(self, state: dict, t: int) -> dict:
        """Generate exogenous information W_{t+1}.

        This represents new information arriving from outside the
        system. It can be:
        - A random sample from a probability distribution
        - A draw from historical data (data-driven simulation)
        - A scenario from a scenario tree
        - A deterministic forecast realization
        """
        pass

    @abstractmethod
    def transition(self, state: dict, decision, exog_info: dict) -> dict:
        """Transition function S_{t+1} = S^M(S_t, x_t, W_{t+1}).

        Computes the next state given the current state, the decision
        made, and the exogenous information that arrived.

        Returns a new state dict with the same structure as build_state().
        """
        pass

    @abstractmethod
    def objective(self, state: dict, decision) -> float:
        """Compute contribution/cost C(S_t, x_t).

        This is the single-period contribution (reward) or cost
        associated with making decision x_t in state S_t.

        For maximization problems, return a positive reward.
        For minimization problems, return a negative cost (or
        negate later in the simulation loop).
        """
        pass

    @abstractmethod
    def get_decision_space(self, state: dict) -> list:
        """Return feasible decisions X_t(S_t).

        The set of feasible decisions may depend on the current state.
        For discrete problems, return a list of all feasible decisions.
        For continuous problems, return bounds or constraints.
        """
        pass
```

### The SDPPolicy Abstract Base Class

`SDPPolicy` represents the policy function X^pi(S_t). All four policy classes (PFA, CFA, VFA, DLA) are implemented as subclasses of this base.

```python
class SDPPolicy(ABC):
    """Base class for policies.

    A policy is a function that maps a state S_t to a decision x_t.
    This is X^pi(S_t) in Powell's notation.

    The four policy classes are all subclasses:
    - PFA: Policy Function Approximation (lookup tables, parametric rules)
    - CFA: Cost Function Approximation (modified optimization)
    - VFA: Value Function Approximation (approximate dynamic programming)
    - DLA: Direct Lookahead (tree search, rollout, stochastic programming)
    """

    @abstractmethod
    def get_decision(self, model: SDPModel, state: dict, t: int):
        """Return decision x_t given current state S_t.

        This is X^pi(S_t) -- the policy function.

        Args:
            model: The SDPModel instance (provides objective, transitions, etc.)
            state: Current state dict
            t: Current time step

        Returns:
            A decision from the feasible set.
        """
        pass

    def update(self, state, decision, new_state, reward):
        """Optional: update policy based on observed transition.

        Used by learning policies (VFA, some PFA variants) to improve
        the policy over time based on experience.

        Args:
            state: State before decision
            decision: Decision taken
            new_state: State after transition
            reward: Observed reward/contribution
        """
        pass
```

---

## The Simulation Loop

The simulation loop is the outer evaluation mechanism. It runs Monte Carlo simulations to estimate the expected cumulative reward of a given policy applied to a given model.

```python
import numpy as np

def simulate(model: SDPModel, policy: SDPPolicy, T: int,
             n_simulations: int = 1000, initial_info: dict = None,
             discount_factor: float = 1.0):
    """Run Monte Carlo simulation to evaluate a policy.

    This is the OUTER loop that evaluates:
        E[ sum_{t=0}^{T-1} gamma^t * C(S_t, X^pi(S_t)) ]

    Args:
        model: The SDPModel defining the problem.
        policy: The SDPPolicy to evaluate.
        T: Number of time steps per episode.
        n_simulations: Number of Monte Carlo replications.
        initial_info: Dict of initial information for build_state().
        discount_factor: Discount factor gamma (default 1.0, undiscounted).

    Returns:
        Dict with 'mean', 'std', 'ci_95', and 'all_rewards'.
    """
    if initial_info is None:
        initial_info = {}

    total_rewards = []

    for sim in range(n_simulations):
        state = model.build_state(initial_info)
        cumulative_reward = 0.0

        for t in range(T):
            # 1. Make decision using policy
            decision = policy.get_decision(model, state, t)

            # 2. Compute single-period contribution
            reward = model.objective(state, decision)
            cumulative_reward += (discount_factor ** t) * reward

            # 3. Generate exogenous information
            exog = model.exogenous_info(state, t)

            # 4. Transition to next state
            new_state = model.transition(state, decision, exog)

            # 5. (Optional) Let the policy learn from the transition
            policy.update(state, decision, new_state, reward)

            state = new_state

        total_rewards.append(cumulative_reward)

    total_rewards = np.array(total_rewards)

    return {
        'mean': np.mean(total_rewards),
        'std': np.std(total_rewards),
        'ci_95': (
            np.percentile(total_rewards, 2.5),
            np.percentile(total_rewards, 97.5)
        ),
        'all_rewards': total_rewards
    }
```

---

## Implementing the Four Policy Classes in Python

### 4.1 PFA Pattern: Policy Function Approximation

PFA policies use a direct mapping from state to decision, typically with tunable parameters. They do not solve any optimization subproblem.

**Threshold Policy for Asset Selling:**

```python
class ThresholdPolicy(SDPPolicy):
    """PFA: Sell if price exceeds threshold.

    This is a parametric policy with one tunable parameter theta
    (the threshold). The policy function is:
        X^pi(S_t) = 'sell' if price_t >= theta, else 'hold'
    """

    def __init__(self, threshold: float):
        self.threshold = threshold  # Tunable parameter theta

    def get_decision(self, model, state, t):
        if state['physical']['price'] >= self.threshold:
            return 'sell'
        return 'hold'
```

**Linear Decision Rule (another PFA variant):**

```python
class LinearPFAPolicy(SDPPolicy):
    """PFA: Linear function of state features.

    X^pi(S_t) = theta_0 + theta_1 * f_1(S_t) + ... + theta_n * f_n(S_t)
    """

    def __init__(self, theta: np.ndarray, feature_fn):
        self.theta = theta          # Parameter vector
        self.feature_fn = feature_fn  # Maps state -> feature vector

    def get_decision(self, model, state, t):
        features = self.feature_fn(state)
        raw_decision = np.dot(self.theta, features)
        # Project onto feasible set
        feasible = model.get_decision_space(state)
        return self._project(raw_decision, feasible)

    def _project(self, value, feasible):
        """Project continuous value onto feasible set."""
        if isinstance(feasible, tuple):  # (low, high) bounds
            return np.clip(value, feasible[0], feasible[1])
        # For discrete feasible sets, find nearest
        return min(feasible, key=lambda x: abs(x - value))
```

### 4.2 CFA Pattern: Cost Function Approximation

CFA policies solve a modified deterministic optimization problem. The modification involves adding tunable correction terms (safety stocks, penalty terms, adjusted parameters) to handle uncertainty.

**Parameterized Inventory Ordering:**

```python
class CFAInventoryPolicy(SDPPolicy):
    """CFA: Order using deterministic model with safety stock parameter.

    Solves a newsvendor-style problem where the order-up-to level is:
        target = E[demand] + theta * std(demand)

    The parameter theta controls the safety stock level and is tuned
    via simulation.
    """

    def __init__(self, safety_stock_factor: float):
        self.theta = safety_stock_factor  # Tunable parameter

    def get_decision(self, model, state, t):
        expected_demand = state['belief']['demand_mean']
        safety_stock = self.theta * state['belief']['demand_std']
        target = expected_demand + safety_stock
        order_qty = max(0, target - state['physical']['inventory'])
        return order_qty
```

**CFA with Optimization Subproblem:**

```python
from scipy.optimize import linprog

class CFAOptimizationPolicy(SDPPolicy):
    """CFA: Solve a modified LP/MIP at each decision point.

    The optimization includes tunable penalty/bonus terms that adjust
    for uncertainty not captured in the deterministic formulation.
    """

    def __init__(self, theta_dict: dict):
        self.theta = theta_dict  # Multiple tunable parameters

    def get_decision(self, model, state, t):
        # Build modified cost vector: original costs + correction terms
        c = model.get_base_costs(state)
        c_modified = c + self.theta.get('cost_adjustment', 0)

        # Build constraints (may also have tunable RHS adjustments)
        A, b = model.get_constraints(state)
        b_modified = b + self.theta.get('rhs_adjustment', 0)

        # Solve modified deterministic optimization
        result = linprog(c_modified, A_ub=A, b_ub=b_modified)
        return result.x
```

### 4.3 VFA Pattern: Value Function Approximation

VFA policies approximate the value function V(S_t) and use it to select decisions that maximize immediate reward plus expected future value.

**Approximate Value Iteration with Lookup Table:**

```python
class VFAPolicy(SDPPolicy):
    """VFA: Use approximate value function to make decisions.

    The decision rule is:
        x_t = argmax_{x in X_t} [ C(S_t, x) + gamma * V_bar(S^M(S_t, x, E[W])) ]

    where V_bar is the value function approximation.
    """

    def __init__(self, value_function, gamma: float = 0.99):
        self.V = value_function  # Approximation architecture
        self.gamma = gamma

    def get_decision(self, model, state, t):
        best_decision = None
        best_value = float('-inf')

        for x in model.get_decision_space(state):
            # Immediate contribution
            immediate = model.objective(state, x)

            # Approximate expected future value using point estimate
            # of exogenous info (or sample average)
            expected_exog = model.get_expected_exogenous(state, t)
            next_state = model.transition(state, x, expected_exog)
            expected_future = self.V.predict(next_state)

            total = immediate + self.gamma * expected_future
            if total > best_value:
                best_value = total
                best_decision = x

        return best_decision

    def update(self, state, decision, new_state, reward):
        """Temporal difference (TD) update for value function.

        V_bar(S_t) <- (1 - alpha) * V_bar(S_t) + alpha * [C_t + gamma * V_bar(S_{t+1})]
        """
        target = reward + self.gamma * self.V.predict(new_state)
        self.V.update(state, target)
```

**Lookup Table Value Function:**

```python
class LookupTableVF:
    """Simple lookup table value function approximation.

    Works for discrete, low-dimensional state spaces.
    Uses exponential smoothing for updates.
    """

    def __init__(self, alpha: float = 0.1, default_value: float = 0.0):
        self.table = {}
        self.alpha = alpha
        self.default = default_value

    def _state_key(self, state):
        """Convert state dict to hashable key."""
        return tuple(sorted(
            (k, tuple(v.items()) if isinstance(v, dict) else v)
            for k, v in state.items()
        ))

    def predict(self, state) -> float:
        key = self._state_key(state)
        return self.table.get(key, self.default)

    def update(self, state, target: float):
        key = self._state_key(state)
        old = self.table.get(key, self.default)
        self.table[key] = (1 - self.alpha) * old + self.alpha * target
```

**Linear Value Function Approximation:**

```python
class LinearVF:
    """Linear value function approximation.

    V_bar(S) = theta^T * phi(S)

    where phi(S) is a feature vector extracted from state S.
    """

    def __init__(self, n_features: int, alpha: float = 0.01):
        self.theta = np.zeros(n_features)
        self.alpha = alpha

    def predict(self, state_features: np.ndarray) -> float:
        return np.dot(self.theta, state_features)

    def update(self, state_features: np.ndarray, target: float):
        prediction = self.predict(state_features)
        error = target - prediction
        self.theta += self.alpha * error * state_features
```

### 4.4 DLA Pattern: Direct Lookahead

DLA policies solve an optimization problem over a future horizon at each decision point. The key idea is receding horizon: solve over the full horizon but only implement the first decision.

**Deterministic Lookahead:**

```python
class DLAPolicy(SDPPolicy):
    """DLA: Solve deterministic optimization over horizon H.

    At each time t, solve:
        max sum_{t'=t}^{t+H} C(S_t', x_t')
        subject to deterministic forecasts for W_{t'+1}

    Then implement ONLY x_t (receding horizon / model predictive control).
    """

    def __init__(self, horizon: int):
        self.H = horizon

    def get_decision(self, model, state, t):
        # Build deterministic forecast over the horizon
        forecast = model.generate_forecast(state, self.H)

        # Solve deterministic optimization using the forecast
        plan = self._solve_deterministic(model, state, forecast)

        # Return ONLY the first decision (receding horizon principle)
        return plan[0]

    def _solve_deterministic(self, model, state, forecast):
        """Solve deterministic optimization (e.g., LP, MIP).

        Uses scipy.optimize, PuLP, Gurobi, or OR-Tools depending
        on problem structure.

        Args:
            model: The SDPModel instance.
            state: Current state.
            forecast: List of dicts with forecasted exogenous info
                      for each period in the horizon.

        Returns:
            List of decisions for each period in the horizon.
        """
        from scipy.optimize import minimize

        def total_cost(decision_vector):
            s = state.copy()
            total = 0.0
            for h in range(self.H):
                x = decision_vector[h]
                total -= model.objective(s, x)  # Negate for minimization
                s = model.transition(s, x, forecast[h])
            return total

        x0 = np.zeros(self.H)
        bounds = [model.get_decision_bounds(state)] * self.H
        result = minimize(total_cost, x0, bounds=bounds)
        return result.x
```

**Stochastic Lookahead (Two-Stage):**

```python
class StochasticLookaheadPolicy(SDPPolicy):
    """DLA: Two-stage stochastic programming lookahead.

    Generates multiple scenarios for future uncertainty and solves
    a stochastic program to find the best first-stage decision.
    """

    def __init__(self, horizon: int, n_scenarios: int = 50):
        self.H = horizon
        self.n_scenarios = n_scenarios

    def get_decision(self, model, state, t):
        # Generate scenarios
        scenarios = [
            model.sample_scenario(state, self.H)
            for _ in range(self.n_scenarios)
        ]

        best_decision = None
        best_value = float('-inf')

        for x in model.get_decision_space(state):
            # Evaluate x across all scenarios
            avg_value = 0.0
            for scenario in scenarios:
                value = self._evaluate_scenario(model, state, x, scenario)
                avg_value += value
            avg_value /= self.n_scenarios

            if avg_value > best_value:
                best_value = avg_value
                best_decision = x

        return best_decision

    def _evaluate_scenario(self, model, state, first_decision, scenario):
        """Evaluate a first-stage decision under a single scenario."""
        s = model.transition(state, first_decision, scenario[0])
        total = model.objective(state, first_decision)
        for h in range(1, self.H):
            x = self._greedy_decision(model, s)
            total += model.objective(s, x)
            s = model.transition(s, x, scenario[h])
        return total

    def _greedy_decision(self, model, state):
        """Simple greedy policy for future stages."""
        best_x = None
        best_r = float('-inf')
        for x in model.get_decision_space(state):
            r = model.objective(state, x)
            if r > best_r:
                best_r = r
                best_x = x
        return best_x
```

---

## Policy Evaluation and Comparison Pattern

A core workflow in SDA is comparing multiple policies on the same model via simulation. Always report confidence intervals, not just point estimates.

```python
def compare_policies(model: SDPModel, policies: dict, T: int,
                     n_sims: int = 1000, initial_info: dict = None):
    """Compare multiple policies via simulation.

    Args:
        model: The SDPModel instance.
        policies: Dict mapping policy name -> SDPPolicy instance.
        T: Number of time steps per episode.
        n_sims: Number of Monte Carlo replications per policy.
        initial_info: Initial state information.

    Returns:
        Dict mapping policy name -> simulation results.
    """
    results = {}

    for name, policy in policies.items():
        result = simulate(model, policy, T, n_sims, initial_info)
        results[name] = result
        print(f"{name}:")
        print(f"  Mean reward:  {result['mean']:.2f}")
        print(f"  Std dev:      {result['std']:.2f}")
        print(f"  95% CI:       [{result['ci_95'][0]:.2f}, {result['ci_95'][1]:.2f}]")

    # Rank policies
    ranked = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    print("\nRanking (best to worst):")
    for rank, (name, result) in enumerate(ranked, 1):
        print(f"  {rank}. {name}: {result['mean']:.2f}")

    return results


# Usage example
policies = {
    'PFA_threshold_50': ThresholdPolicy(threshold=50),
    'PFA_threshold_60': ThresholdPolicy(threshold=60),
    'CFA_safety_1.0': CFAInventoryPolicy(safety_stock_factor=1.0),
    'CFA_safety_1.5': CFAInventoryPolicy(safety_stock_factor=1.5),
    'VFA_lookup': VFAPolicy(LookupTableVF(alpha=0.1)),
    'DLA_horizon_5': DLAPolicy(horizon=5),
}

results = compare_policies(model, policies, T=100, n_sims=1000)
```

---

## Parameter Tuning for PFA/CFA

PFA and CFA policies have tunable parameters theta that must be optimized via stochastic search (since the objective is an expectation estimated by simulation).

```python
from itertools import product

def tune_policy_grid(model, policy_class, param_grid: dict,
                     T: int, n_sims: int = 100):
    """Grid search to tune policy parameters theta.

    Args:
        model: The SDPModel instance.
        policy_class: The policy class to instantiate.
        param_grid: Dict mapping param name -> list of values to try.
        T: Time horizon.
        n_sims: Simulations per parameter setting (lower for speed).

    Returns:
        Tuple of (best_params, best_performance, all_results).
    """
    best_params = None
    best_performance = float('-inf')
    all_results = []

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in product(*values):
        params = dict(zip(keys, combo))
        policy = policy_class(**params)
        result = simulate(model, policy, T, n_sims)
        all_results.append((params, result))

        if result['mean'] > best_performance:
            best_performance = result['mean']
            best_params = params

    print(f"Best parameters: {best_params}")
    print(f"Best mean reward: {best_performance:.2f}")
    return best_params, best_performance, all_results


def tune_policy_bayesian(model, policy_class, param_bounds: dict,
                         T: int, n_sims: int = 100, n_iterations: int = 50):
    """Bayesian optimization for policy parameter tuning.

    Uses Optuna for sample-efficient optimization.

    Args:
        model: The SDPModel instance.
        policy_class: The policy class to instantiate.
        param_bounds: Dict mapping param name -> (low, high) bounds.
        T: Time horizon.
        n_sims: Simulations per evaluation.
        n_iterations: Number of Bayesian optimization iterations.

    Returns:
        Tuple of (best_params, best_performance).
    """
    import optuna

    def objective(trial):
        params = {}
        for name, (low, high) in param_bounds.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high)

        policy = policy_class(**params)
        result = simulate(model, policy, T, n_sims)
        return result['mean']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_iterations)

    print(f"Best parameters: {study.best_params}")
    print(f"Best mean reward: {study.best_value:.2f}")
    return study.best_params, study.best_value
```

---

## Belief State Implementation Patterns

Belief states capture what the agent knows about uncertain quantities. They are updated via Bayesian updating as new observations arrive.

### Bayesian Updating for Normal-Normal Model

When observations are normally distributed with known variance and the unknown parameter is the mean:

```python
class NormalBelief:
    """Belief about an unknown mean with known observation variance.

    Conjugate Normal-Normal model:
        Prior:      mu ~ N(mu_0, sigma_0^2)
        Likelihood: y | mu ~ N(mu, sigma_obs^2)
        Posterior:  mu | y ~ N(mu_n, sigma_n^2)

    This is the foundation for many learning problems including
    multi-armed bandits, dynamic pricing, and Bayesian regression.
    """

    def __init__(self, prior_mean: float, prior_var: float,
                 obs_var: float):
        self.mu = prior_mean      # Posterior mean (initially prior mean)
        self.var = prior_var      # Posterior variance (initially prior var)
        self.obs_var = obs_var    # Known observation variance
        self.n_obs = 0            # Number of observations seen

    def update(self, observation: float):
        """Update belief with a new observation (Bayesian update).

        Uses precision-weighted combination:
            precision_new = precision_prior + precision_obs
            mean_new = (precision_prior * mean_prior + precision_obs * obs) / precision_new
        """
        precision_prior = 1.0 / self.var
        precision_obs = 1.0 / self.obs_var
        new_precision = precision_prior + precision_obs

        self.mu = (precision_prior * self.mu +
                   precision_obs * observation) / new_precision
        self.var = 1.0 / new_precision
        self.n_obs += 1

    def sample(self) -> float:
        """Draw a sample from the current posterior (for Thompson Sampling)."""
        return np.random.normal(self.mu, np.sqrt(self.var))

    def confidence_interval(self, level: float = 0.95) -> tuple:
        """Return a credible interval for the unknown mean."""
        from scipy.stats import norm
        z = norm.ppf((1 + level) / 2)
        half_width = z * np.sqrt(self.var)
        return (self.mu - half_width, self.mu + half_width)
```

### Thompson Sampling for Multi-Armed Bandits

Thompson Sampling is a PFA policy that uses belief states to balance exploration and exploitation.

```python
class ThompsonSamplingPolicy(SDPPolicy):
    """Thompson Sampling for Bernoulli multi-armed bandits.

    Maintains Beta posterior for each arm's success probability.
    At each step, samples from each posterior and selects the arm
    with the highest sample.

    This is a PFA policy where the belief state IS the state.
    """

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.alphas = np.ones(n_arms)  # Beta prior: successes + 1
        self.betas = np.ones(n_arms)   # Beta prior: failures + 1

    def get_decision(self, model, state, t):
        """Sample from each arm's posterior and pick the best."""
        samples = [
            np.random.beta(a, b)
            for a, b in zip(self.alphas, self.betas)
        ]
        return np.argmax(samples)

    def update(self, state, decision, new_state, reward):
        """Update Beta posterior for the chosen arm."""
        arm = decision
        self.alphas[arm] += reward        # Increment successes
        self.betas[arm] += (1 - reward)   # Increment failures

    def get_means(self) -> np.ndarray:
        """Return posterior mean for each arm."""
        return self.alphas / (self.alphas + self.betas)

    def get_upper_bounds(self, quantile: float = 0.975) -> np.ndarray:
        """Return upper credible bound for each arm (for UCB comparison)."""
        from scipy.stats import beta
        return np.array([
            beta.ppf(quantile, a, b)
            for a, b in zip(self.alphas, self.betas)
        ])
```

### Knowledge Gradient for Offline Learning

```python
class KnowledgeGradientPolicy(SDPPolicy):
    """Knowledge Gradient policy for offline learning problems.

    Selects the arm that maximizes the expected improvement in the
    value of the best arm after one more observation.

    KG_x = E[ max_x' mu_{n+1,x'} | S_n, measure x ] - max_x' mu_{n,x'}
    """

    def __init__(self, beliefs: list):
        self.beliefs = beliefs  # List of NormalBelief objects

    def get_decision(self, model, state, t):
        n_arms = len(self.beliefs)
        current_best = max(b.mu for b in self.beliefs)
        kg_values = []

        for arm in range(n_arms):
            kg = self._compute_kg(arm, current_best)
            kg_values.append(kg)

        return np.argmax(kg_values)

    def _compute_kg(self, arm, current_best):
        """Compute Knowledge Gradient value for an arm.

        Uses the analytical formula for the Normal-Normal case.
        """
        belief = self.beliefs[arm]
        sigma_tilde = np.sqrt(belief.var - (belief.var ** 2) /
                              (belief.var + belief.obs_var))
        if sigma_tilde < 1e-10:
            return 0.0

        z = -(abs(belief.mu - current_best)) / sigma_tilde
        from scipy.stats import norm
        kg = sigma_tilde * (z * norm.cdf(z) + norm.pdf(z))
        return kg

    def update(self, state, decision, new_state, reward):
        """Update belief for the measured arm."""
        self.beliefs[decision].update(reward)
```

---

## Worked Example: Energy Storage Problem

This is a complete implementation of the energy storage problem, one of the canonical examples from Powell's framework. The problem involves deciding how much energy to charge or discharge from a battery given fluctuating prices and renewable supply.

```python
import numpy as np
from typing import List, Tuple

# --- Model ---

class EnergyStorageModel(SDPModel):
    """Energy storage model with price uncertainty.

    State: (storage_level, price)
    Decision: charge/discharge amount (continuous)
    Exogenous: next price (random walk or mean-reverting)
    Objective: maximize revenue from buying low / selling high
    """

    def __init__(self, max_storage: float = 100.0,
                 max_charge_rate: float = 20.0,
                 max_discharge_rate: float = 20.0,
                 efficiency: float = 0.9,
                 price_mean: float = 50.0,
                 price_std: float = 10.0,
                 mean_reversion: float = 0.1):
        self.max_storage = max_storage
        self.max_charge = max_charge_rate
        self.max_discharge = max_discharge_rate
        self.efficiency = efficiency
        self.price_mean = price_mean
        self.price_std = price_std
        self.mean_reversion = mean_reversion

    def build_state(self, info: dict) -> dict:
        return {
            'physical': {
                'storage': info.get('initial_storage', self.max_storage / 2),
            },
            'other': {
                'price': info.get('initial_price', self.price_mean),
            }
        }

    def exogenous_info(self, state: dict, t: int) -> dict:
        """Mean-reverting price process with noise."""
        current_price = state['other']['price']
        noise = np.random.normal(0, self.price_std)
        new_price = (current_price +
                     self.mean_reversion * (self.price_mean - current_price) +
                     noise)
        new_price = max(0, new_price)  # Price cannot be negative
        return {'price': new_price}

    def transition(self, state: dict, decision: float,
                   exog_info: dict) -> dict:
        """Update storage level and price.

        decision > 0: charging (buying energy)
        decision < 0: discharging (selling energy)
        """
        storage = state['physical']['storage']

        if decision >= 0:
            # Charging: energy in * efficiency
            new_storage = storage + decision * self.efficiency
        else:
            # Discharging: energy out directly
            new_storage = storage + decision  # decision is negative

        new_storage = np.clip(new_storage, 0, self.max_storage)

        return {
            'physical': {'storage': new_storage},
            'other': {'price': exog_info['price']}
        }

    def objective(self, state: dict, decision: float) -> float:
        """Revenue from energy transaction.

        Selling (decision < 0): revenue = |decision| * price
        Buying (decision > 0): cost = decision * price (negative revenue)
        """
        price = state['other']['price']
        return -decision * price  # Negative decision (sell) yields positive revenue

    def get_decision_space(self, state: dict) -> list:
        """Return list of feasible charge/discharge amounts."""
        storage = state['physical']['storage']
        # Max we can discharge (sell)
        max_sell = min(self.max_discharge, storage)
        # Max we can charge (buy)
        max_buy = min(self.max_charge,
                      (self.max_storage - storage) / self.efficiency)
        # Discretize for simplicity
        decisions = np.linspace(-max_sell, max_buy, 21)
        return decisions.tolist()

    def get_decision_bounds(self, state: dict) -> Tuple[float, float]:
        storage = state['physical']['storage']
        max_sell = min(self.max_discharge, storage)
        max_buy = min(self.max_charge,
                      (self.max_storage - storage) / self.efficiency)
        return (-max_sell, max_buy)

    def generate_forecast(self, state: dict, horizon: int) -> list:
        """Generate deterministic price forecast (for DLA)."""
        forecasts = []
        price = state['other']['price']
        for h in range(horizon):
            price = price + self.mean_reversion * (self.price_mean - price)
            forecasts.append({'price': price})
        return forecasts

    def get_expected_exogenous(self, state: dict, t: int) -> dict:
        """Expected exogenous info (for VFA)."""
        current_price = state['other']['price']
        expected_price = (current_price +
                          self.mean_reversion *
                          (self.price_mean - current_price))
        return {'price': expected_price}


# --- Policies for Energy Storage ---

class BuyLowSellHighPolicy(SDPPolicy):
    """PFA: Buy below threshold, sell above threshold."""

    def __init__(self, buy_threshold: float, sell_threshold: float):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def get_decision(self, model, state, t):
        price = state['other']['price']
        storage = state['physical']['storage']

        if price <= self.buy_threshold:
            max_buy = min(model.max_charge,
                          (model.max_storage - storage) / model.efficiency)
            return max_buy
        elif price >= self.sell_threshold:
            max_sell = min(model.max_discharge, storage)
            return -max_sell
        else:
            return 0.0


class CFAEnergyPolicy(SDPPolicy):
    """CFA: Target storage level based on price relative to mean."""

    def __init__(self, theta_target: float, theta_rate: float):
        self.theta_target = theta_target  # Target fill fraction [0,1]
        self.theta_rate = theta_rate      # Rate adjustment factor

    def get_decision(self, model, state, t):
        price = state['other']['price']
        storage = state['physical']['storage']

        # Adjust target based on price: store more when cheap, less when expensive
        price_ratio = price / model.price_mean
        adjusted_target = self.theta_target * model.max_storage / price_ratio

        gap = adjusted_target - storage
        decision = np.clip(gap * self.theta_rate,
                           -min(model.max_discharge, storage),
                           min(model.max_charge,
                               (model.max_storage - storage) /
                               model.efficiency))
        return decision


class VFAEnergyPolicy(SDPPolicy):
    """VFA: Approximate value function with linear features."""

    def __init__(self, model: EnergyStorageModel, alpha: float = 0.05,
                 gamma: float = 0.99):
        self.gamma = gamma
        # Features: [1, storage, price, storage*price, storage^2, price^2]
        self.theta = np.zeros(6)
        self.alpha = alpha
        self.model = model

    def _features(self, state: dict) -> np.ndarray:
        s = state['physical']['storage'] / self.model.max_storage
        p = state['other']['price'] / self.model.price_mean
        return np.array([1.0, s, p, s * p, s ** 2, p ** 2])

    def _value(self, state: dict) -> float:
        return np.dot(self.theta, self._features(state))

    def get_decision(self, model, state, t):
        best_decision = None
        best_total = float('-inf')

        for x in model.get_decision_space(state):
            immediate = model.objective(state, x)
            expected_exog = model.get_expected_exogenous(state, t)
            next_state = model.transition(state, x, expected_exog)
            future = self._value(next_state)
            total = immediate + self.gamma * future

            if total > best_total:
                best_total = total
                best_decision = x

        return best_decision

    def update(self, state, decision, new_state, reward):
        target = reward + self.gamma * self._value(new_state)
        prediction = self._value(state)
        error = target - prediction
        features = self._features(state)
        self.theta += self.alpha * error * features


class DLAEnergyPolicy(SDPPolicy):
    """DLA: Deterministic lookahead for energy storage."""

    def __init__(self, horizon: int = 10):
        self.H = horizon

    def get_decision(self, model, state, t):
        forecast = model.generate_forecast(state, self.H)

        # Simple heuristic lookahead: find min/max price periods
        prices = [f['price'] for f in forecast]
        current_price = state['other']['price']

        future_max = max(prices) if prices else current_price
        future_min = min(prices) if prices else current_price

        storage = state['physical']['storage']

        # If current price is high relative to future, sell
        if current_price > future_max * 0.95:
            max_sell = min(model.max_discharge, storage)
            return -max_sell
        # If current price is low relative to future, buy
        elif current_price < future_min * 1.05:
            max_buy = min(model.max_charge,
                          (model.max_storage - storage) / model.efficiency)
            return max_buy
        else:
            return 0.0


# --- Run the full comparison ---

def run_energy_storage_experiment():
    """Complete experiment comparing all four policy classes."""

    model = EnergyStorageModel(
        max_storage=100.0,
        max_charge_rate=20.0,
        max_discharge_rate=20.0,
        efficiency=0.9,
        price_mean=50.0,
        price_std=10.0,
        mean_reversion=0.1
    )

    T = 50       # 50 time steps per episode
    n_sims = 500  # Monte Carlo replications

    # Define policies
    policies = {
        'PFA_conservative': BuyLowSellHighPolicy(
            buy_threshold=40, sell_threshold=60
        ),
        'PFA_aggressive': BuyLowSellHighPolicy(
            buy_threshold=45, sell_threshold=55
        ),
        'CFA_target_0.5': CFAEnergyPolicy(
            theta_target=0.5, theta_rate=0.3
        ),
        'CFA_target_0.7': CFAEnergyPolicy(
            theta_target=0.7, theta_rate=0.5
        ),
        'VFA_linear': VFAEnergyPolicy(model, alpha=0.05, gamma=0.99),
        'DLA_horizon_10': DLAEnergyPolicy(horizon=10),
    }

    # Compare policies
    results = compare_policies(model, policies, T, n_sims)

    # Tune PFA parameters
    print("\nTuning PFA thresholds...")
    best_params, best_perf, _ = tune_policy_grid(
        model, BuyLowSellHighPolicy,
        param_grid={
            'buy_threshold': [35, 40, 42, 45],
            'sell_threshold': [55, 58, 60, 65]
        },
        T=T, n_sims=200
    )
    print(f"Tuned PFA: {best_params} -> {best_perf:.2f}")

    return results


if __name__ == '__main__':
    results = run_energy_storage_experiment()
```

---

## Key Libraries and Tools

The following libraries are commonly used when implementing SDA patterns in Python:

### Core Numerics
- **numpy**: Array operations, random number generation, statistics
- **scipy**: Scientific computing utilities
- **scipy.optimize**: Optimization solvers for CFA and DLA subproblems (`linprog`, `minimize`, `milp`)
- **scipy.stats**: Probability distributions for belief updates and sampling

### Optimization Solvers (for CFA and DLA)
- **PuLP**: Open-source LP/MIP modeling library (good for medium-scale problems)
- **Gurobi** (via `gurobipy`): High-performance commercial LP/MIP/QP solver
- **Google OR-Tools** (`ortools`): Open-source constraint programming and routing
- **CVXPY**: Convex optimization modeling (useful for quadratic objectives)

### Machine Learning (for VFA and advanced PFA)
- **PyTorch**: Neural network value function approximation (deep RL)
- **JAX**: Differentiable programming for gradient-based VFA updates
- **scikit-learn**: Feature engineering, regression models for VFA

### Parameter Tuning (for PFA and CFA)
- **Optuna**: Bayesian hyperparameter optimization
- **Ax** (from Meta): Adaptive experimentation platform
- **GPyOpt**: Gaussian process-based Bayesian optimization

### Simulation and Environment Interface
- **Gymnasium** (OpenAI Gym): Standard RL environment interface (compatible with SDA)
- **SimPy**: Discrete-event simulation (for complex operational models)

### Official and Community Repositories
- **wbpowell328/stochastic-optimization**: Powell's original Python library with implementations of all examples from the textbook
- **djanka2/stochastic-optimization**: Fork with the `SDPModel` and `SDPPolicy` abstract base classes used throughout this skill
- **Peymankor/seqdec_powell_repo**: Community notebooks with worked examples and visualizations

---

## Integration with Gymnasium/OpenAI Gym

SDPModel can be wrapped as a Gymnasium environment, enabling use with standard RL libraries (Stable-Baselines3, RLlib, CleanRL).

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SDAGymWrapper(gym.Env):
    """Wrap an SDPModel as a Gymnasium environment.

    This allows any SDA model to be used with standard RL libraries
    such as Stable-Baselines3, RLlib, or CleanRL.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, sdp_model: SDPModel, T: int,
                 obs_low: np.ndarray = None, obs_high: np.ndarray = None,
                 action_low: float = -1.0, action_high: float = 1.0,
                 n_obs_dims: int = 2):
        super().__init__()
        self.model = sdp_model
        self.T = T
        self.t = 0
        self.state = None

        # Define observation and action spaces
        if obs_low is None:
            obs_low = np.full(n_obs_dims, -np.inf)
        if obs_high is None:
            obs_high = np.full(n_obs_dims, np.inf)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([action_low]),
            high=np.array([action_high]),
            dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """Convert state dict to observation array."""
        return np.array([
            self.state['physical'].get('storage', 0),
            self.state['other'].get('price', 0),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.state = self.model.build_state({})
        self.t = 0
        return self._get_obs(), {}

    def step(self, action):
        """Execute one step: decision -> reward -> exogenous -> transition."""
        # Convert action array to scalar decision
        decision = float(action[0])

        # Compute reward
        reward = self.model.objective(self.state, decision)

        # Generate exogenous information
        exog = self.model.exogenous_info(self.state, self.t)

        # Transition
        self.state = self.model.transition(self.state, decision, exog)
        self.t += 1

        # Check termination
        terminated = (self.t >= self.T)
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}


# Usage with Stable-Baselines3
def train_rl_agent(sdp_model, T, total_timesteps=100000):
    """Train an RL agent on an SDA model using Stable-Baselines3."""
    from stable_baselines3 import PPO

    env = SDAGymWrapper(sdp_model, T)
    agent = PPO('MlpPolicy', env, verbose=1)
    agent.learn(total_timesteps=total_timesteps)
    return agent


# Wrap the trained RL agent as an SDPPolicy
class RLAgentPolicy(SDPPolicy):
    """Wrap a trained RL agent as an SDPPolicy for comparison."""

    def __init__(self, agent, wrapper: SDAGymWrapper):
        self.agent = agent
        self.wrapper = wrapper

    def get_decision(self, model, state, t):
        self.wrapper.state = state
        obs = self.wrapper._get_obs()
        action, _ = self.agent.predict(obs, deterministic=True)
        return float(action[0])
```

---

## Testing and Validation Patterns

Rigorous testing is essential for SDA implementations. The following patterns ensure correctness and reliability.

### 1. Compare Against Known Baselines

Always compute a **posterior optimal** (perfect information) benchmark. This upper bound tells you how much room for improvement exists.

```python
def compute_posterior_optimal(model, T, n_sims=1000):
    """Compute posterior optimal: best reward with perfect foresight.

    This is NOT achievable in practice (requires seeing the future)
    but provides an upper bound for policy evaluation.
    """
    optimal_rewards = []

    for sim in range(n_sims):
        # Generate full trajectory of exogenous information
        state = model.build_state({})
        exog_sequence = []
        s = state
        for t in range(T):
            exog = model.exogenous_info(s, t)
            exog_sequence.append(exog)
            s = model.transition(s, 0, exog)  # Dummy transition for price

        # Now optimize with perfect knowledge of exog_sequence
        best_reward = _optimize_with_perfect_info(model, state, exog_sequence)
        optimal_rewards.append(best_reward)

    return {
        'mean': np.mean(optimal_rewards),
        'std': np.std(optimal_rewards),
        'ci_95': (np.percentile(optimal_rewards, 2.5),
                  np.percentile(optimal_rewards, 97.5))
    }
```

### 2. Use Confidence Intervals

Never rely on point estimates alone. Always report confidence intervals and use enough simulations for statistical significance.

```python
def is_significantly_better(result_a, result_b, alpha=0.05):
    """Test if policy A is significantly better than policy B.

    Uses a two-sample t-test on the simulation results.
    """
    from scipy.stats import ttest_ind

    t_stat, p_value = ttest_ind(
        result_a['all_rewards'],
        result_b['all_rewards'],
        alternative='greater'
    )
    return p_value < alpha, p_value
```

### 3. Test Multiple Problem Variants

Following Powell's approach with the five energy storage variants, always test on multiple configurations to check robustness.

```python
def sensitivity_analysis(model_class, model_configs: list,
                         policies: dict, T: int, n_sims: int = 500):
    """Run sensitivity analysis across multiple problem variants.

    Args:
        model_class: The SDPModel class.
        model_configs: List of dicts with model parameters.
        policies: Dict of policy name -> SDPPolicy.
        T: Time horizon.
        n_sims: Simulations per (model, policy) pair.

    Returns:
        DataFrame with results for all combinations.
    """
    rows = []

    for config_idx, config in enumerate(model_configs):
        model = model_class(**config)
        for policy_name, policy in policies.items():
            result = simulate(model, policy, T, n_sims)
            rows.append({
                'config': config_idx,
                'config_params': str(config),
                'policy': policy_name,
                'mean_reward': result['mean'],
                'std_reward': result['std'],
                'ci_lower': result['ci_95'][0],
                'ci_upper': result['ci_95'][1],
            })

    import pandas as pd
    return pd.DataFrame(rows)
```

### 4. Validate Transition Function

Ensure the transition function conserves resources and respects constraints.

```python
def validate_model(model: SDPModel, n_steps: int = 100):
    """Basic validation checks for an SDPModel implementation."""
    state = model.build_state({})

    for t in range(n_steps):
        decisions = model.get_decision_space(state)
        assert len(decisions) > 0, f"No feasible decisions at t={t}"

        decision = np.random.choice(decisions)
        reward = model.objective(state, decision)
        assert np.isfinite(reward), f"Non-finite reward at t={t}"

        exog = model.exogenous_info(state, t)
        new_state = model.transition(state, decision, exog)

        # Check state structure is preserved
        assert set(new_state.keys()) == set(state.keys()), \
            f"State keys changed at t={t}"

        state = new_state

    print("Model validation passed.")
```

---

## References

### GitHub Repositories
- **wbpowell328/stochastic-optimization**: Warren Powell's original Python library with implementations of all textbook examples. Contains model definitions, policy implementations, and simulation infrastructure for asset selling, inventory management, energy storage, and more.
- **djanka2/stochastic-optimization**: Fork providing the `SDPModel` and `SDPPolicy` abstract base classes used as the architectural foundation in this skill. Offers a clean, extensible interface for implementing new problems.
- **Peymankor/seqdec_powell_repo**: Community-maintained Jupyter notebooks reproducing key examples from Powell's textbook. Includes visualizations, step-by-step walkthroughs, and additional experiments.

### Textbooks and Papers
- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions*. Wiley. The foundational reference for the four-policy-class framework and all modeling patterns.
- Powell, W.B. (2022). *Sequential Decision Analytics and Modeling*. Now Publishers. Companion text with Python module implementations and worked examples covering the full modeling pipeline.

### Key Concepts Cross-Reference
- **SDPModel** maps to the universal model (S, x, W, S^M, C) from the modeling framework
- **SDPPolicy** maps to X^pi(S_t), the policy function
- **simulate()** implements the outer Monte Carlo loop for policy evaluation
- **compare_policies()** implements the experimental framework for policy comparison
- **tune_policy_grid/bayesian()** implements stochastic search for PFA/CFA parameter tuning
- **NormalBelief** implements the Bayesian belief state for learning problems
- **SDAGymWrapper** bridges the SDA framework with the RL ecosystem
