# SDA Skill: Belief States, Learning Under Uncertainty, and Exploration/Exploitation

## 1. Overview

Sequential Decision Analytics (SDA) recognizes that most real-world decisions must be made with incomplete knowledge. You rarely know the true demand distribution, the actual click-through rate, or the real efficacy of a treatment. The SDA framework handles this systematically through the concept of the **belief state**.

**Core principles:**

- The **belief state** B_t captures everything we currently know or believe about uncertain quantities at time t.
- The full state variable is S_t = (R_t, B_t), combining the **physical state** R_t (inventory levels, resource positions, bank balances) with the **belief state** B_t (what we believe about unknowns).
- **Learning** is the process of updating B_t as new information W_t arrives after each decision.
- The **exploration/exploitation dilemma** is fundamental: should we gather more information (explore) or use what we already know to get the best immediate outcome (exploit)?
- Ignoring the belief state collapses many sequential problems into myopic ones, forfeiting the long-term value of information.

**Why this matters for implementation:** Every time you build an SDA model, you must ask: "What don't I know? How do my decisions affect what I learn? And how does what I learn affect future decisions?" If the answer to any of these is nontrivial, you need an explicit belief state.

---

## 2. What Is a Belief State?

### Definition

A belief state B_t is a **probability distribution** (or sufficient statistics thereof) that describes what we currently believe about unknown quantities. It is not a single guess; it is a structured representation of uncertainty.

### Forms of the Belief State

| Unknown quantity | Belief state B_t | Sufficient statistics |
|---|---|---|
| Unknown scalar parameter mu | Posterior distribution over mu | (mu_bar_t, sigma_sq_t) -- posterior mean and variance |
| Unknown probability p | Beta posterior | Beta(alpha_t, beta_t) |
| Unknown rate lambda | Gamma posterior | Gamma(a_t, b_t) |
| Unknown categorical distribution | Dirichlet posterior | Dirichlet(alpha_t_1, ..., alpha_t_K) |
| Unknown function f(x) | Gaussian process posterior | Mean function m_t(x), covariance function k_t(x, x') |
| Unknown value function V(s) | Bayesian ADP beliefs | (V_bar_t(s), sigma_sq_t(s)) for each state s |

### Why the Belief State Must Be Part of the State Variable

B_t is PART of the state variable S_t = (R_t, B_t). This is not optional. If you drop B_t from the state, the process is **no longer Markov**: future transitions depend on historical observations that are not captured in R_t alone. Including B_t restores the Markov property and makes the problem well-posed for dynamic programming or policy search.

### The Bayesian Framework

The belief state follows a Bayesian updating cycle:

1. **Prior:** B_0 represents initial beliefs before any data is observed. This encodes domain expertise, historical data, or uninformative priors.
2. **Decision:** At time t, choose action x_t based on current state S_t = (R_t, B_t).
3. **Observation:** After acting, observe new information W_{t+1}. Critically, the decision x_t often determines WHAT we observe (e.g., choosing to test drug A means we observe drug A's outcome, not drug B's).
4. **Likelihood:** The probability model P(W_{t+1} | x_t, theta) connecting observations to unknowns.
5. **Posterior update:** B_{t+1} = Bayes(B_t, W_{t+1}). The transition function for B_t IS Bayesian updating.

### Concrete Examples

**Drug efficacy (Beta-Bernoulli):**
- Unknown: true success probability p of a treatment.
- Belief state: B_t = Beta(alpha_t, beta_t), where alpha_t counts observed successes and beta_t counts observed failures.
- Prior: B_0 = Beta(1, 1) (uniform) or Beta(alpha_0, beta_0) based on prior studies.
- After observing a success: B_{t+1} = Beta(alpha_t + 1, beta_t).
- After observing a failure: B_{t+1} = Beta(alpha_t, beta_t + 1).
- Posterior mean: alpha_t / (alpha_t + beta_t).

**Mean demand (Normal-Normal):**
- Unknown: true mean demand mu.
- Belief state: B_t = Normal(mu_bar_t, sigma_sq_t / n_t), where mu_bar_t is the running sample mean and n_t is the effective sample count.
- After observing demand y_{t+1}: update via standard Normal-Normal conjugate formulas.
- Posterior mean: weighted average of prior mean and new observation.
- Posterior variance: shrinks with each observation.

**Click-through rates for multiple ads (multi-arm):**
- Unknown: true CTR p_a for each ad a in {1, ..., A}.
- Belief state: B_t = {Beta(alpha_{t,a}, beta_{t,a}) for each a}.
- Only the arm/ad that is SHOWN gets its belief updated.
- This is the classic multi-armed bandit setup.

---

## 3. The Transition Function for Beliefs

### General Form

The belief state transition follows:

```
B_{t+1} = f^B(B_t, x_t, W_{t+1})
```

Three inputs matter:
- **B_t**: current beliefs (the prior for this step).
- **x_t**: the decision, which determines what information we receive.
- **W_{t+1}**: the exogenous information/observation that arrives.

The decision x_t is critical because it controls the **information channel**. Choosing arm 3 in a bandit means you learn about arm 3, not about arms 1 or 2. Choosing to survey region A means you learn about region A's geology, not region B's.

### Conjugate Priors (Closed-Form Updates)

When the prior and likelihood form a conjugate pair, the posterior has the same distributional family as the prior, and updates are available in closed form.

**Normal-Normal (unknown mean, known variance sigma_sq_W):**
```
Prior:       mu ~ Normal(mu_bar_t, sigma_sq_t)
Likelihood:  W_{t+1} | mu ~ Normal(mu, sigma_sq_W)
Posterior:   mu | W_{t+1} ~ Normal(mu_bar_{t+1}, sigma_sq_{t+1})

Where:
  sigma_sq_{t+1} = 1 / (1/sigma_sq_t + 1/sigma_sq_W)
  mu_bar_{t+1}   = sigma_sq_{t+1} * (mu_bar_t / sigma_sq_t + W_{t+1} / sigma_sq_W)
```

**Beta-Bernoulli (binary outcomes):**
```
Prior:       p ~ Beta(alpha_t, beta_t)
Likelihood:  W_{t+1} | p ~ Bernoulli(p)
Posterior:   p | W_{t+1} ~ Beta(alpha_t + W_{t+1}, beta_t + 1 - W_{t+1})
```

**Gamma-Poisson (count data, unknown rate):**
```
Prior:       lambda ~ Gamma(a_t, b_t)
Likelihood:  W_{t+1} | lambda ~ Poisson(lambda)
Posterior:   lambda | W_{t+1} ~ Gamma(a_t + W_{t+1}, b_t + 1)
```

**Dirichlet-Multinomial (categorical outcomes):**
```
Prior:       (p_1, ..., p_K) ~ Dirichlet(alpha_t_1, ..., alpha_t_K)
Likelihood:  W_{t+1} | p ~ Multinomial(p)
Posterior:   Dirichlet(alpha_t_1 + c_1, ..., alpha_t_K + c_K)
             where c_k = count of category k in the observation
```

### Non-Conjugate Cases

When conjugacy does not hold, exact Bayesian updating is typically intractable. Common approaches:

- **Particle filters:** Represent B_t as a weighted set of samples (particles). Update weights via the likelihood, resample periodically.
- **Markov Chain Monte Carlo (MCMC):** Draw samples from the posterior. Expensive per update but flexible.
- **Variational inference:** Approximate the posterior with a tractable family. Fast but introduces approximation error.
- **Discretization:** Bin the belief space and maintain a probability over bins. Works for low-dimensional unknowns.
- **Moment matching:** After an exact update, project back onto a tractable family by matching first and second moments.

---

## 4. Exploration vs. Exploitation

### The Dilemma

This is the central tension in learning problems:

- **Exploitation:** Choose the action with the highest expected reward given current beliefs B_t. This is the greedy choice: argmax_a E[reward(a) | B_t].
- **Exploration:** Choose an action specifically to learn more, even if its expected immediate reward is lower. The value comes from improved future decisions.
- **The trade-off:** Every exploratory action has an opportunity cost (foregone immediate reward), but every exploitative action has an information cost (foregone learning).

Maximizing long-term cumulative reward requires balancing both. The right balance depends on the time horizon, the number of alternatives, the current state of knowledge, and the cost of suboptimal decisions.

### Pure Exploitation (Greedy Policy)

```
X^Greedy(S_t) = argmax_a  E[reward(a) | B_t]
```

- Always picks the action that looks best RIGHT NOW.
- **Works well when:** prior beliefs are already accurate, learning happens fast through passive observation, the time horizon is very long relative to the learning period, or there are few alternatives.
- **Fails when:** initial beliefs are poor or heavily biased, there are many options to learn about, the time horizon is short, or the cost of not learning is high.
- Common failure mode: locking onto a suboptimal action early because it happened to produce good initial outcomes by chance.

### 4.1 Epsilon-Greedy

```
X^EpsGreedy(S_t) =
  random action uniformly from A     with probability epsilon
  argmax_a E[reward(a) | B_t]        with probability 1 - epsilon
```

**Properties:**
- Simple to implement.
- Crude: does not use belief state information to guide exploration.
- Explores ALL actions equally, including clearly bad ones.
- Parameter epsilon can decay over time: epsilon_t = epsilon_0 / (1 + t/tau) to shift from exploration to exploitation.
- A useful baseline but rarely the best strategy.

### 4.2 Boltzmann Exploration (Softmax)

```
P(action = a | S_t) = exp(Q_t(a) / tau) / sum_a' exp(Q_t(a') / tau)
```

**Properties:**
- Temperature tau controls exploration intensity. High tau -> near-uniform random. Low tau -> near-greedy.
- Better than epsilon-greedy: exploits value differences (prefers high-value actions even when exploring).
- Still does NOT consider uncertainty -- two actions with the same mean but very different variances get the same probability.
- tau can be annealed over time.

### 4.3 Upper Confidence Bounding (UCB)

```
X^UCB(S_t) = argmax_a [ mu_bar_{t,a} + c * sqrt(log(t) / n_{t,a}) ]
```

Where mu_bar_{t,a} is the estimated mean reward for arm a, n_{t,a} is the number of times arm a has been tried, and c is an exploration constant.

**Properties:**
- Implements the principle of **optimism in the face of uncertainty**: act as if the unknown parameters are at the upper end of their confidence interval.
- Automatically explores actions with high uncertainty (large sqrt(log(t)/n_{t,a})) OR high estimated value (large mu_bar_{t,a}).
- In the SDA policy taxonomy, UCB is a **CFA-style policy**: it is a parameterized optimization where the exploration constant c is the tunable parameter.
- Theoretical regret guarantees: O(sqrt(K * T * log(T))) for K arms over T rounds.
- Variants: UCB1, UCB-V (variance-aware), KL-UCB, LinUCB (contextual).

### 4.4 Thompson Sampling

```
Algorithm:
1. For each action a, sample theta_tilde_a ~ B_t(a)    [sample from posterior]
2. Choose X(S_t) = argmax_a  reward(theta_tilde_a)      [act greedily on sample]
```

**Properties:**
- Naturally balances exploration and exploitation through the mechanics of posterior sampling: uncertain actions have high-variance posteriors, so they occasionally produce optimistic samples that cause them to be selected.
- As the posterior concentrates (we learn), exploration decreases automatically.
- In the SDA policy taxonomy, Thompson sampling is a **PFA** (Policy Function Approximation): the decision is a deterministic function of the (randomly sampled) state.
- Elegant, easy to implement for conjugate models, and often competitive with or superior to UCB.
- Works well even when the model is somewhat misspecified.
- Theoretical regret bounds comparable to UCB for many settings.

**Implementation for Beta-Bernoulli bandits:**
```python
# B_t = {(alpha_a, beta_a) for each arm a}
for each arm a:
    theta_sample_a = np.random.beta(alpha_a, beta_a)
chosen_arm = argmax(theta_sample_a)
# Observe reward, update B_{t+1} for chosen arm
```

### 4.5 Knowledge Gradient (KG)

The Knowledge Gradient is a key contribution of Powell and collaborators. It directly computes the **value of information** from each measurement and chooses the action that maximizes it.

```
X^KG(S_t) = argmax_a  nu^KG(S_t, a)
```

Where nu^KG(S_t, a) is the **marginal value of measuring action a**, defined as:

```
nu^KG(S_t, a) = E[ max_{a'} mu_bar_{t+1,a'} | x_t = a, B_t ] - max_{a'} mu_bar_{t,a'}
```

This is the expected improvement in our ability to make a good decision (as measured by the best posterior mean) after observing one sample from arm a.

**Properties:**
- **One-step optimal:** if you have exactly one measurement remaining, KG selects the action that maximizes the expected value of the final decision. This is provably optimal for the single-remaining-measurement case.
- For multiple remaining measurements, KG is a heuristic (myopic with respect to the value of information), but often an excellent one.
- Extends naturally to **correlated beliefs**: if learning about arm a also tells us about arm b (through correlated priors), KG accounts for this.
- Works in both **online** settings (rewards during learning matter) and **offline** settings (only the final recommendation matters). In online settings, combine KG with exploitation: X = argmax_a [mu_bar_{t,a} + (T - t) * nu^KG(S_t, a)] or similar.
- In the SDA policy taxonomy, KG is a **hybrid PFA/VFA**: it computes a value-of-information quantity and uses it to make decisions.

**When KG works best:**
- Small to moderate number of alternatives.
- Each measurement is expensive (limited budget).
- Beliefs are correlated across alternatives.
- Offline learning / ranking and selection.

**Citation:** Frazier, Powell, Dayanik (2008, 2009).

### 4.6 Gittins Index

```
X^Gittins(S_t) = argmax_a  gamma(B_{t,a})
```

Where gamma(B_{t,a}) is the Gittins index for arm a given its current belief state.

**Properties:**
- Provides the **provably optimal** solution for the multi-armed bandit with independent arms and geometric discounting (discount factor beta < 1).
- The Gittins index for an arm is the "retirement reward" that makes you indifferent between playing that arm forever and receiving the retirement reward.
- Each arm's index depends ONLY on its own belief state, enabling decomposition of a high-dimensional problem.
- In the SDA policy taxonomy, the Gittins index is a **VFA** (Value Function Approximation): the index IS the value function for each arm's sub-problem.

**Limitations -- the Gittins index breaks down for:**
- Correlated arms (learning about one arm tells you about others).
- Finite horizon (no geometric discounting).
- Contextual features (state-dependent rewards).
- Arms with switching costs.
- Non-stationary environments.

When these limitations apply, use Thompson sampling, UCB, or KG instead.

---

## 5. Types of Learning Problems

### 5.1 Pure Learning (Offline)

- **Goal:** Identify the best option using a fixed budget of N experiments.
- The decisions made during learning carry no inherent reward or cost; only the **final recommendation** matters.
- Examples: ranking and selection, A/B testing design, drug screening in preclinical trials, materials discovery, simulation optimization.
- **Policy evaluation metric:** Opportunity cost of the final recommendation = E[mu^* - mu_{a_recommended}].
- Knowledge Gradient is particularly well-suited here.
- The exploration/exploitation trade-off simplifies to pure exploration with an information budget.

### 5.2 Online Learning

- Learning happens WHILE making consequential decisions with real costs/rewards.
- Every action generates both a reward (or cost) and information.
- Must balance learning and earning simultaneously.
- Examples: clinical trials (patients receive treatment), ad placement (revenue depends on which ad is shown), dynamic pricing (revenue depends on price set), inventory management with unknown demand.
- **Policy evaluation metric:** Cumulative regret = sum of [mu^* - mu_{a_t}] over all periods.
- Thompson sampling and UCB are natural fits here.

### 5.3 Model-Based Learning

- The model structure is known (e.g., demand = f(price, season) + noise), but some parameters are unknown.
- Learning = estimating the unknown parameters within the known model.
- The belief state tracks uncertainty about model parameters.
- Both the belief state B_t and the physical state R_t affect decisions.
- Example: inventory management where the demand distribution family is known (Normal) but the mean and variance are unknown. B_t = beliefs about (mu, sigma_sq). R_t = current inventory level. The stocking decision depends on BOTH.

### 5.4 Model-Free Learning

- No explicit model of the environment is assumed.
- Learning a value function V(s) or Q(s,a), or a policy pi(s), directly from data.
- Q-learning, SARSA, policy gradient methods, actor-critic.
- There is STILL an implicit belief state: the current weights/parameters of the neural network or function approximator encode what has been learned. The weights theta_t of a Q-network represent B_t in this setting.
- Exploration remains essential: epsilon-greedy and Boltzmann exploration are common in this setting; more principled methods (posterior sampling over network weights, bootstrapped DQN) are active research areas.

---

## 6. Bayesian Exploration for Approximate Dynamic Programming

Powell and Ryzhov developed a framework that brings Bayesian belief states into Approximate Dynamic Programming (ADP):

**Core idea:** Maintain Bayesian beliefs not just about environment parameters, but about the **value function** V(s) itself.

```
B_t = { (V_bar_t(s), sigma_sq_t(s)) for each state s }
```

Where V_bar_t(s) is the current estimate of the value of state s, and sigma_sq_t(s) is the uncertainty in that estimate.

**How it works:**
1. When ADP visits a state s and computes a sample of V(s), this is treated as a noisy observation.
2. The belief about V(s) is updated via Bayesian updating (Normal-Normal conjugacy).
3. When choosing which state to visit/explore next, use the **value of information** -- prefer states where sigma_sq_t(s) is large and where better knowledge of V(s) would most improve the overall policy.

**Why this matters:**
- Standard ADP/RL explores via epsilon-greedy or random restarts, which is undirected.
- Bayesian ADP uses belief-state-driven exploration, which is **directed**: it knows WHERE the uncertainty is and explores there.
- This can dramatically improve sample efficiency in large state spaces.
- Connects the exploration/exploitation framework for bandits to the full sequential decision problem.

**Citation:** Ryzhov, Mes, Powell, van den Berg (2019). "Bayesian Exploration for Approximate Dynamic Programming." Operations Research.

---

## 7. The Knowledge Gradient in Detail

### Mathematical Formulation (Independent Normal Priors)

Setup: K alternatives with unknown true values mu_1, ..., mu_K. Current beliefs:
```
mu_a ~ Normal(mu_bar_{t,a}, sigma_sq_{t,a})  independently for each a
```

Measurement of alternative a yields observation:
```
y = mu_a + epsilon,  epsilon ~ Normal(0, sigma_sq_epsilon)
```

After measuring alternative a, the posterior for arm a updates (Normal-Normal conjugacy):
```
mu_bar_{t+1,a} = (sigma_sq_{t,a} * y + sigma_sq_epsilon * mu_bar_{t,a}) / (sigma_sq_{t,a} + sigma_sq_epsilon)
```

The KG value for measuring alternative a is:
```
nu^KG_a = E[ max_{a'} mu_bar_{t+1,a'} ] - max_{a'} mu_bar_{t,a'}
```

Since only arm a's belief changes, this simplifies to a one-dimensional computation involving the normal CDF. Let sigma_tilde = sigma_sq_{t,a} / sqrt(sigma_sq_{t,a} + sigma_sq_epsilon). Then:

```
nu^KG_a = sigma_tilde * f( -|delta_a| / sigma_tilde )
```

Where delta_a = mu_bar_{t,a} - max_{a' != a} mu_bar_{t,a'} and f(z) = z * Phi(z) + phi(z) with Phi and phi being the standard normal CDF and PDF respectively.

### Extension to Correlated Beliefs

When beliefs about different alternatives are correlated (e.g., alternatives share features, or learning about one product informs beliefs about related products):

```
mu ~ Normal(mu_bar_t, Sigma_t)   [joint prior, Sigma_t is K x K]
```

Measuring alternative a updates the ENTIRE covariance matrix:
```
Sigma_{t+1} = Sigma_t - (Sigma_t e_a e_a^T Sigma_t) / (e_a^T Sigma_t e_a + sigma_sq_epsilon)
```

The correlated KG accounts for the fact that measuring arm a updates beliefs about ALL arms. This makes KG especially powerful when alternatives are related.

### Computational Considerations

- For independent beliefs with K arms: O(K log K) per decision (sorting step).
- For correlated beliefs: O(K^2) per decision (matrix update).
- For large K: approximations exist (e.g., sample-based KG, sparse covariance).
- In practice, KG is most valuable when K is moderate (tens to hundreds, not millions).

### Comparison with Alternatives

| Method | Uses uncertainty? | Correlated beliefs? | Theoretical guarantee | Computational cost |
|---|---|---|---|---|
| Epsilon-greedy | No | No | Sublinear regret | O(1) |
| UCB | Yes (frequentist) | No (standard) | O(sqrt(KT log T)) | O(K) |
| Thompson sampling | Yes (Bayesian) | Yes (naturally) | Bayesian regret bounds | O(K * sample cost) |
| Knowledge gradient | Yes (Bayesian) | Yes (explicitly) | One-step optimal | O(K log K) to O(K^2) |
| Gittins index | Yes (Bayesian) | No | Optimal (restricted case) | O(K * index computation) |

---

## 8. Practical Guidance

### Decision Tree for Choosing an Exploration Strategy

1. **Is learning even necessary?** If your initial model is well-calibrated from historical data and the environment is stationary, a greedy policy may suffice. Test this first.

2. **Is this offline (pure exploration) or online (explore while earning)?**
   - Offline: Use Knowledge Gradient or a budget-allocation formulation.
   - Online: Continue to step 3.

3. **How many alternatives/arms?**
   - Small (< 100): Thompson sampling or Knowledge Gradient.
   - Large (100-10000): Thompson sampling or UCB (lower per-step cost).
   - Very large / continuous: Contextual bandits (LinUCB, neural Thompson sampling).

4. **Are alternatives correlated?**
   - Yes: Knowledge Gradient with correlated beliefs, or Thompson sampling with a joint posterior.
   - No: Any method works; Thompson sampling is a strong default.

5. **How expensive is each measurement?**
   - Cheap (millions of trials): UCB or Thompson sampling.
   - Expensive (tens to hundreds of trials): Knowledge Gradient.

### Rules of Thumb

- **Start with Thompson sampling.** It is simple, effective, robust to model misspecification, and works in both online and offline settings. It is the best "default" exploration strategy.
- **Use Knowledge Gradient when measurements are expensive** and you can afford the computational cost. It excels in ranking-and-selection problems with correlated beliefs.
- **Use UCB for streaming/online settings** where you want frequentist guarantees and simplicity.
- **Do not forget the greedy baseline.** In many practical problems, especially those with long horizons or fast passive learning, a simple greedy policy works well enough. Always benchmark against it.
- **Consider hybrid strategies:** Explore early (large epsilon, high temperature, or pure KG), then switch to exploitation as beliefs stabilize. A practical heuristic: if posterior standard deviation is less than some fraction of the posterior mean for all arms, stop exploring.
- **Monitor convergence of beliefs.** Plot sigma_sq_t over time. If uncertainty is not decreasing, your model or learning rate may be misconfigured.

### Common Pitfalls

- **Forgetting the belief state in the state variable.** If you implement a "learning" system but do not track B_t explicitly, your policy cannot reason about the value of information.
- **Over-exploring.** Exploration has a cost. In online settings with short horizons, too much exploration destroys cumulative performance.
- **Under-exploring.** Locking onto the first arm that looks good (the greedy trap). This is especially dangerous with noisy observations and many arms.
- **Ignoring correlations.** If alternatives share structure (e.g., similar drugs, related products), treating them as independent wastes information.
- **Using the wrong prior.** An overconfident prior (too-small variance) can prevent exploration. An uninformative prior when you actually have domain knowledge wastes early measurements.

---

## 9. Connection to the Four Policy Classes

Powell's four classes of policies provide a taxonomy for ALL decision-making strategies. Each exploration method maps naturally:

### Thompson Sampling -> PFA (Policy Function Approximation)
- The policy is a deterministic function of the (stochastic) sampled state.
- Given a sample theta_tilde from the posterior, the decision is deterministic: pick the best arm under theta_tilde.
- The randomness comes from sampling, not from the policy itself.

### UCB -> CFA (Cost Function Approximation)
- UCB modifies the objective function by adding an exploration bonus.
- The decision solves: argmax_a [mu_bar_{t,a} + c * uncertainty_bonus(a)].
- The exploration constant c is a tunable parameter, characteristic of CFA policies.
- The base optimization (argmax) is the "cost function" being approximated.

### Gittins Index -> VFA (Value Function Approximation)
- The Gittins index for each arm IS the value of playing that arm optimally into the future.
- The index decomposes the multi-arm value function into per-arm value functions.
- Choosing the arm with the highest index is equivalent to choosing the arm with the highest value-to-go.

### Monte Carlo Tree Search (MCTS) -> DLA (Direct Lookahead Approximation)
- MCTS builds a stochastic lookahead tree, sampling future outcomes.
- Exploration within the tree (e.g., UCT = UCB applied to tree nodes) balances exploring new branches vs. deepening known-good branches.
- The entire tree search is an approximate solution to a stochastic lookahead model.

### Knowledge Gradient -> Hybrid PFA/VFA
- KG computes a value-of-information quantity (VFA-like: it estimates the value of a measurement).
- But the decision rule is a direct function of the current state and the KG values (PFA-like: a policy function maps state to action).
- This hybrid nature is typical of policies derived from one-step value-of-information analysis.

---

## 10. Key Citations and Further Reading

### Primary References

- **Frazier, P., Powell, W.B., Dayanik, S. (2008).** "A Knowledge-Gradient Policy for Sequential Information Collection." *SIAM Journal on Control and Optimization*, 47(5), 2410-2439. The foundational paper on the Knowledge Gradient for independent normal priors.

- **Frazier, P., Powell, W.B., Dayanik, S. (2009).** "The Knowledge-Gradient Policy for Correlated Normal Beliefs." *INFORMS Journal on Computing*, 21(4), 599-613. Extension to correlated beliefs across alternatives.

- **Ryzhov, I.O., Mes, M.R.K., Powell, W.B., van den Berg, G.J. (2019).** "Bayesian Exploration for Approximate Dynamic Programming." *Operations Research*, 67(1), 198-214. Applying Bayesian belief states to value function learning in ADP.

- **Powell, W.B. (2022).** *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions.* Wiley. Chapters 7-11 cover belief states, Bayesian learning, and exploration in depth within the SDA framework.

- **Powell, W.B.** "The Knowledge Gradient for Optimal Learning." Wiley Encyclopedia of Operations Research and Management Science. A concise overview of KG theory and applications.

### Supplementary References

- **Gittins, J., Glazebrook, K., Weber, R. (2011).** *Multi-Armed Bandit Allocation Indices.* 2nd edition. Wiley. The definitive reference on Gittins indices.

- **Russo, D.J., Van Roy, B., Kazerouni, A., Osband, I., Wen, Z. (2018).** "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning*, 11(1), 1-96. Comprehensive tutorial on Thompson sampling.

- **Lattimore, T., Szepesvari, C. (2020).** *Bandit Algorithms.* Cambridge University Press. Rigorous treatment of bandit algorithms including UCB and Thompson sampling.

- **DeGroot, M.H. (1970).** *Optimal Statistical Decisions.* McGraw-Hill. Classic reference on Bayesian decision theory and the value of information.
