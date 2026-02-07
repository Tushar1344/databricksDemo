# SDA Frequently Asked Questions

A collection of deep conceptual questions and answers about Warren B. Powell's Sequential Decision Analytics framework. These go beyond definitions into the "why" behind design choices.

---

## Q1: Why is exogenous information (W_t) separate from state? Can't we expand system boundaries to make it endogenous?

### Short Answer

You **can** expand system boundaries — and when you do, what was exogenous becomes endogenous state. But Powell deliberately keeps them separate for three powerful reasons: controllability, tractability, and clean simulation design.

### 1. The Controllability Boundary

The defining characteristic of W_t is: **the decision-maker cannot influence it**.

- Weather, competitor actions, patient biology, market prices — these happen regardless of what you decide
- State S_t, by contrast, is shaped by your past decisions via the transition function
- If you expand the boundary to model weather as endogenous state, you now need a transition function *for weather* — but you have no decision variable that affects it. You've added model complexity for zero decision-making benefit

### 2. Computational and Modeling Tractability

This is the practical killer argument:

- If you absorb all exogenous processes into the state, **your state space explodes**. The global economy, weather systems, every other agent's internal state — all become "state variables"
- Powell's framework says: model only what you *need* to make decisions and compute transitions
- W_t arrives from *outside* the model. You only need to model its **effect on your state** (via the transition function), not its generating process
- **Example:** An inventory manager doesn't need to model the entire economy. They need demand D_t (exogenous) and how it affects inventory: `R_{t+1} = R_t + x_t - D_t`

### 3. The Separation Enables Clean Simulation

The framework's power comes from this factorization:

```
S_{t+1} = S^M(S_t, x_t, W_{t+1})
```

- **S_t** = what you know and control
- **x_t** = what you decide
- **W_{t+1}** = what the world throws at you

This separation lets you:
- **Swap uncertainty models** without changing the decision model (plug in different W_t distributions)
- **Replay historical data** as W_t sequences
- **Run Monte Carlo** by sampling W_t independently from the policy

If W_t were absorbed into S_t, you'd lose this clean separation between "your model" and "the world's randomness."

### When You SHOULD Expand the Boundary

Sometimes the boundary should move:

- **If your decisions affect the information you receive** — e.g., choosing to measure something (active learning). Then the *observation* is partly endogenous, but the *underlying truth* remains exogenous. Powell handles this through belief states: your decision x_t affects which W_{t+1} you *observe*, but not the underlying reality
- **In multi-agent problems** — another agent's decisions look exogenous to you, but are endogenous to them. Game theory and multi-agent SDA explicitly model this boundary choice
- **If forecasts are part of your state** — a price forecast might be exogenous information, but *incorporating it into your belief state* makes it part of S_t. This is exactly the belief state B_t component

### The Deep Insight

Powell's framework is a **modeling choice**, not an ontological claim. The boundary between "state" and "exogenous information" is chosen by the modeler to produce the most useful, tractable model. The key test is:

> *"Does including this in my state help me make better decisions, or does it just add dimensions I can't act on?"*

- If you can't act on it → keep it exogenous (W_t)
- If it affects your decisions and you need to track it → put it in state (S_t)

This is analogous to **system boundaries in thermodynamics** or **control volumes in fluid dynamics** — the boundary is a modeling choice that trades off completeness against tractability.

### Citations
- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization*, Ch. 9 (Modeling Uncertainty), Ch. 11 (State Variables).
- Powell, W.B. (2022). *Sequential Decision Analytics and Modeling*, Ch. 1 and Ch. 7.

---

## Q2: Why does Powell say "Bellman's equation is the least useful of the four classes of policies"?

### Context

This is a provocative claim given that Bellman's equation is the foundation of dynamic programming and reinforcement learning — two of the most celebrated frameworks in sequential decision-making.

### Powell's Argument

1. **Curse of dimensionality:** Bellman's equation requires computing V(S) for every state S. For high-dimensional state spaces (inventory across 1000 products, fleet of 10,000 vehicles), this is computationally intractable even with approximation.

2. **Most real-world problems are solved with PFA or CFA:** Industry practitioners overwhelmingly use rule-based policies (PFAs) or parameterized optimization models (CFAs). These methods don't require a value function at all.

3. **Instability of value function approximation:** The "deadly triad" in RL (function approximation + bootstrapping + off-policy learning) makes VFA methods fragile in practice. Deep RL breakthroughs (Atari, Go) required enormous computational resources and careful engineering.

4. **CFA handles high-dimensional resource allocation naturally:** When you have an existing optimization model (LP, MIP), parameterizing it (CFA) scales far better than trying to approximate a value function over a massive state space.

5. **DLA often outperforms VFA for planning problems:** When good forecasts are available, a deterministic or stochastic lookahead can make excellent decisions without ever estimating long-term value.

### What Powell Does NOT Mean

- He does NOT mean VFA is useless — it's essential for some problems (e.g., games, low-dimensional control)
- He does NOT mean Bellman's equation is mathematically wrong — it's a correct identity
- He DOES mean that the research community's disproportionate focus on VFA/Bellman methods has caused them to overlook PFA, CFA, and DLA, which are often simpler and more effective

### The Energy Storage Proof

In Powell's canonical energy storage illustration, five variants of the same problem each favor a different policy class. VFA wins on exactly one variant. This is empirical evidence that no single class dominates — and VFA is not special.

### Citations
- Powell, W.B. (2022). *RLSO*, Chs. 14-18 (detailed treatment of VFA methods and their limitations).
- Powell, W.B. (2019). "A unified framework for stochastic optimization." *EJOR*, 275(3), 795-821.

---

## Q3: How is SDA different from Reinforcement Learning?

### Short Answer

SDA **subsumes** RL. Reinforcement learning is primarily concerned with one policy class (VFA — value function approximation), while SDA covers all four classes and provides a universal modeling framework.

### Detailed Comparison

| Dimension | Reinforcement Learning | Sequential Decision Analytics |
|-----------|----------------------|------------------------------|
| **Policy methods** | Primarily VFA (Q-learning, DQN, actor-critic, policy gradient) | All four: PFA, CFA, VFA, DLA |
| **Modeling** | Environment provides (s, a, r, s') tuples; model often implicit | Explicit five-element model: S_t, x_t, W_t, S^M, objective |
| **Notation** | (s, a, r) from MDP tradition | (S_t, x_t, C_t) from optimization tradition |
| **State design** | Often given by the environment | Deliberate modeling choice including belief states |
| **Uncertainty** | Implicit in environment dynamics | Explicitly modeled as W_t |
| **Optimization focus** | Learn value function or policy from experience | Choose the RIGHT policy class, then optimize within it |
| **Typical applications** | Games, robotics, simulated environments | Transportation, energy, health, supply chain, finance |
| **Industry adoption** | Growing but limited to specific domains | CFA/PFA widely used in industry (often without the label) |

### What RL Gets Right
- Powerful when the environment is a black box (no model available)
- Deep RL scales to complex perception tasks (images, language)
- Principled exploration through value-based methods

### What RL Misses (per Powell)
- Ignores CFA entirely — the most common industrial approach
- Treats PFA as "too simple" rather than as a legitimate policy class
- Overemphasizes model-free methods when models are often available
- Uses a single notation (MDP) that obscures connections to optimization, control, and stochastic programming

### The Reconciliation
SDA doesn't reject RL — it places it in context. When someone says "use RL," they typically mean "use VFA" (Q-learning, DQN, PPO, etc.). SDA asks: "Is VFA the right tool here, or would PFA, CFA, or DLA work better?"

### Citations
- Powell, W.B. (2022). *RLSO*, Ch. 1 ("The Challenges of Sequential Decision Problems").
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*. (For the RL perspective.)

---

## Q4: When should I use a Cost Function Approximation (CFA) vs. a Value Function Approximation (VFA)?

### Use CFA When:
- You already have a deterministic optimization model (LP, MIP, shortest path, scheduling model)
- The state/decision space is high-dimensional (hundreds or thousands of variables)
- Domain experts have intuition about how uncertainty affects the optimal solution (e.g., "add safety stock," "increase reserve margin")
- You need interpretable, auditable decisions
- Computational speed at decision time matters (solve one optimization, not iterate a value function)
- Industry practice already uses parameterized heuristics (formalize and tune them)

### Use VFA When:
- The state space is low-to-moderate dimensional
- There's clear value-of-state structure (the value of being in state S is meaningful and smooth)
- The problem has infinite horizon with stationary structure
- You need to capture long-horizon consequences that a simple parameterization can't express
- The problem is naturally a "learning" problem (bandits, exploration)

### Use Both (Hybrid) When:
- CFA for the main optimization, VFA for terminal value or constraint pricing
- VFA to set CFA parameters (meta-learning)
- Complex resource allocation with both planning and learning components

### Powell's Rule of Thumb
> Start with the simplest policy class that might work (often PFA or CFA). Only add complexity (VFA, DLA) if the simpler approach fails to capture important structure.

### Citations
- Powell, W.B. (2022). *RLSO*, Chs. 12-13 (PFA and CFA) vs. Chs. 14-18 (VFA).
- Ghadimi, S. & Powell, W.B. (2024). "Stochastic search for a parametric cost function approximation." *EJOR*.

---

## Q5: What makes a state variable "Markovian" and how do I know if my state definition is complete?

### The Markov Property

A state S_t is Markovian if:

> The future evolution of the system depends only on S_t (and future decisions and exogenous information), NOT on the history (S_0, x_0, W_1, ..., S_{t-1}).

In other words: **the state contains everything you need to model forward**. History doesn't matter beyond what's captured in the current state.

### How to Test Your State Definition

**Test 1: The Decision Test**
- Given only S_t, can you make a decision? If you find yourself wanting to "look back" at previous states or observations, your state is incomplete.

**Test 2: The Transition Test**
- Given (S_t, x_t, W_{t+1}), can you compute S_{t+1} deterministically? If not, something is missing from S_t.

**Test 3: The Value Test**
- Do two trajectories that arrive at the same S_t have the same expected future value? If not, S_t doesn't capture enough information.

### Common Sources of Non-Markov Behavior (and Fixes)

| Symptom | Missing State Component | Fix |
|---------|------------------------|-----|
| Need to remember past observations | Belief state B_t | Add sufficient statistics (mean, variance, counts) |
| Decision depends on trend | Lagged values or trend estimate | Add S_{t-1} or trend variable to state |
| Decision depends on time | Time index | Add t to state |
| Decision depends on previous action | Last action taken | Add x_{t-1} to state |
| Decision depends on cumulative reward | Running total | Add cumulative reward to state |

### The Belief State is the Most Commonly Forgotten Component

If you're learning about the environment (unknown demand, unknown efficacy, uncertain parameters), you MUST include your beliefs about those unknowns as part of S_t. Otherwise:
- The problem appears non-Markov (same physical state, different histories, different optimal actions)
- The fix: S_t = (R_t, B_t) where B_t is the belief state

### Practical Advice

Don't over-specify the state (curse of dimensionality). Include only what's needed for decisions and transitions. The art of SDA modeling is finding the **minimal sufficient state**.

### Citations
- Powell, W.B. (2022). *RLSO*, Ch. 11 (State Variables).
- Powell, W.B. (2022). *SDAM*, Ch. 7 (State Variables Revisited).

---

## Q6: How do the "15 fragmented communities" relate to the four policy classes?

### The Problem

Powell identifies 15+ academic communities that all study sequential decision problems but use different names, notations, and methods:

| Community | Typical Notation | Preferred Policy Class | What They Often Miss |
|-----------|-----------------|----------------------|---------------------|
| Reinforcement Learning | (s, a, r, s') | VFA (Q-learning, DQN) | CFA, simple PFA |
| Dynamic Programming | V(s), Bellman equation | VFA (exact or approximate) | CFA |
| Stochastic Programming | scenario trees, recourse | DLA (stochastic) | PFA, CFA |
| Optimal Control | x(k+1) = f(x,u,w) | DLA (MPC) and VFA (HJB) | CFA |
| Simulation Optimization | black-box f(θ) | PFA (parameter search) | VFA, DLA |
| Bandits / Online Learning | (K arms, regret) | PFA (UCB, Thompson) | CFA, DLA |
| Bayesian Optimization | GP surrogate, acquisition | PFA (knowledge gradient) | CFA, DLA |
| Markov Decision Processes | (S, A, P, R, γ) | VFA (value/policy iteration) | CFA |
| Model Predictive Control | receding horizon | DLA (deterministic) | PFA, CFA, VFA |
| Robust Optimization | min-max, uncertainty sets | DLA (worst-case) | PFA, CFA |
| Decision Analysis | decision trees | DLA (simple tree) | CFA, VFA at scale |
| Operations Research | LP/MIP + heuristics | CFA (ad hoc) | Formal VFA |
| Machine Learning | loss, SGD, neural nets | VFA (deep RL) | CFA |
| Economics | structural estimation | PFA (estimated rules) | CFA, VFA |
| Management Science | analytical models | PFA (closed-form) | VFA, DLA |

### The Insight

Every community has **blind spots** — policy classes they don't teach, don't consider, or actively dismiss. SDA's contribution is showing that **you need all four classes** and providing a universal language to bridge these communities.

### The Practical Consequence

When you read a paper from one community, ask: "Which policy class did they use? Which ones did they NOT consider?" Often, a simpler policy class from a different community would work just as well or better.

### Citations
- Powell, W.B. (2019). "A unified framework for stochastic optimization." *EJOR*.
- Powell, W.B. "From the Jungle of Stochastic Optimization to Sequential Decision Analytics." castle.princeton.edu/jungle/

---

## Q7: Is the SDA framework only theoretical, or is it used in industry?

### Industry Adoption

SDA is not just academic — the key insight is that **industry has been using SDA methods (especially PFA and CFA) for decades**, just without the formal framework:

| Industry | SDA Method in Use | Example |
|----------|-------------------|---------|
| **Trucking/Logistics** | CFA | Optimal Dynamics (Powell's company) uses SDA for fleet optimization |
| **Power Systems** | CFA | Unit commitment solved with parameterized deterministic models — standard practice at ISOs worldwide |
| **Supply Chain** | PFA | (s, S) and (r, q) inventory policies used universally |
| **Finance** | PFA + DLA | Trading rules (PFA) + portfolio optimization with forecasts (DLA) |
| **Tech/Advertising** | PFA | Thompson sampling and UCB for ad placement at scale (Google, Meta) |
| **Navigation** | DLA | Google Maps = deterministic shortest path lookahead |
| **Healthcare** | PFA | Clinical guidelines as policy function approximations |
| **Water Management** | CFA | Reservoir management with parameterized rule curves |

### What SDA Adds to Industry Practice

The gap between industry practice and SDA theory is the **tuning step**. Industry uses PFAs and CFAs but often with ad-hoc parameter choices. SDA formalizes the objective function for tuning:

```
max_θ  E[ Σ_t C(S_t, X^π(S_t | θ)) ]
```

This turns heuristic parameter selection into a rigorous stochastic optimization problem, yielding 20-30% improvements in computational experiments.

### Citations
- Ghadimi & Powell (2024). Energy storage CFA results: 22-30% cost improvement.
- Optimal Dynamics: optimaldynamics.com (industry application of SDA to trucking).
- Powell, W.B. (2022). *RLSO*, Ch. 13 (CFA industry examples).

---

## Q8: What are common misunderstandings about state variables that lead to poor modeling?

State variable design is the **single most consequential modeling decision** in SDA. Get it wrong and no policy — however sophisticated — can compensate. Here are the most common misunderstandings, organized from most to least frequent.

---

### Misunderstanding 1: Forgetting the Belief State

**The mistake:** Defining S_t as only physical/tangible quantities and ignoring what the agent *believes* about unknown quantities.

**Why it's damaging:**
- If there are unknown parameters you're learning about (demand distribution, treatment efficacy, click-through rates), your beliefs about them **must** be in the state
- Without B_t, two situations with identical physical state but different learning histories look the same to the policy — but they aren't. The optimal action for "I've seen this drug work 8/10 times" is completely different from "I've never tried this drug"
- Technically, omitting B_t violates the Markov property: the optimal decision depends on history that isn't captured in the state

**The fix:** S_t = (R_t, B_t) where:
- R_t = physical/resource state (inventory, location, energy stored)
- B_t = belief state (posterior distributions, sufficient statistics, confidence intervals)

**Example:**
```
BAD:   S_t = (inventory_level)
       → Same state whether demand is well-understood or completely unknown

GOOD:  S_t = (inventory_level, demand_mean_estimate, demand_variance_estimate)
       → Policy can order conservatively when uncertain, aggressively when confident
```

**How to detect it:** If your policy "should" behave differently for two scenarios that have the same state representation, you're missing state variables — usually beliefs.

---

### Misunderstanding 2: Confusing "Observable" with "State"

**The mistake:** Defining the state as "everything I can observe right now."

**Why it's wrong:**
- State is not "what I can see" — it's "what I need to model forward and make decisions"
- Some state variables are derived, not directly observed (e.g., a running average, a trend estimate, a Bayesian posterior)
- Some observations are exogenous information W_t (new data arriving), not state. Seeing today's price is an observation; your *model of price dynamics* is state

**The fix:** State = minimal set of variables such that, given S_t, you can:
1. Make a decision
2. Compute the transition to S_{t+1} (given x_t and W_{t+1})
3. Evaluate the objective function

**Example:**
```
CONFUSED:  "My state is the current stock price"
           → But your decision also depends on your position, risk tolerance,
             and beliefs about volatility — those are all state

CLEAR:     S_t = (price_t, shares_held_t, volatility_estimate_t, time_remaining)
```

---

### Misunderstanding 3: Over-Specifying the State (Kitchen Sink)

**The mistake:** Throwing everything into the state "just in case" — full order history, every sensor reading, the complete weather forecast, every customer interaction.

**Why it's damaging:**
- **Curse of dimensionality:** VFA methods scale poorly with state dimension. A state with 1000 components makes value function approximation nearly impossible
- **Overfitting:** Policies trained on high-dimensional states may memorize training scenarios rather than learn generalizable structure
- **Computational cost:** Every additional state variable multiplies the space that must be explored
- **Obscures structure:** The modeler can't see the essential dynamics buried under noise variables

**The fix:** Apply the **minimal sufficiency test** — for each candidate state variable, ask:
- Does removing it change the optimal decision in any scenario? If no → remove it
- Does removing it make the transition function non-deterministic (given W_t)? If no → remove it
- Can it be derived from other state variables? If yes → remove it

**Example:**
```
OVER-SPECIFIED:  S_t = (inventory, last_30_days_demand, supplier_lead_times,
                        weather_forecast, GDP_growth, competitor_prices,
                        day_of_week, holiday_flag, ...)
                 → 50+ dimensional state; impossible to learn a value function

MINIMAL:         S_t = (inventory, demand_mean_estimate, demand_std_estimate,
                        pending_orders)
                 → 4 variables capturing essential dynamics
```

**Powell's guidance:** "The art of SDA modeling is finding the *minimal sufficient state*." Start small, add variables only when they demonstrably improve policy performance.

---

### Misunderstanding 4: Under-Specifying the State (Non-Markov Trap)

**The mistake:** Leaving out variables that the optimal decision depends on, making the problem appear non-Markov.

**Symptoms:**
- Your policy performs inconsistently — same state, different outcomes
- You find yourself wanting to "look back" at past decisions or observations
- The transition function seems to have hidden randomness beyond W_t

**Common missing components:**

| Symptom | What's Missing | Fix |
|---------|---------------|-----|
| Decision depends on momentum/trend | Lagged values or trend variable | Add ΔS_{t-1} or trend estimate |
| Decision depends on "how long we've been here" | Duration/time-in-state | Add counter or elapsed time |
| Decision depends on previous action | Last action taken | Add x_{t-1} to state |
| Decision depends on budget spent so far | Cumulative expenditure | Add running total |
| Decision depends on commitments made | Pending orders/contracts | Add pipeline state |
| Different outcomes for same physical state | Belief state | Add B_t (see Misunderstanding 1) |

**Example:**
```
UNDER-SPECIFIED:  S_t = (current_price)
                  → Can't distinguish "price rising from 50" from "price falling from 50"
                  → Policy can't capture momentum-based strategies

SUFFICIENT:       S_t = (current_price, price_change_last_period)
                  → Now trending up and trending down are different states
```

---

### Misunderstanding 5: Treating State as Static Design Rather Than Dynamic

**The mistake:** Defining the state once at the start and never revisiting as you learn about the problem.

**Why it's wrong:**
- State design is iterative. You discover missing components when policies behave unexpectedly
- As you try different policy classes, you may need different state representations (VFA may need aggregated features; PFA may use raw values)
- The "right" state depends on the policy class you're using

**The fix:** Treat state design as part of the modeling loop:
1. Define initial state → implement policy → evaluate
2. If performance is poor or inconsistent → diagnose: is the state complete?
3. Add/remove variables → re-evaluate
4. Repeat

---

### Misunderstanding 6: Confusing Pre-Decision and Post-Decision State

**The mistake:** Not distinguishing between the state *before* and *after* a decision is made, leading to incorrect value function training.

**Why it matters:**
- **Pre-decision state** S_t: the state before you act. Your policy maps this to a decision
- **Post-decision state** S_t^x: the state after you decide but before new information arrives. S_t^x = S^M_x(S_t, x_t)
- **Post-information state** S_{t+1}: after W_{t+1} arrives. S_{t+1} = S^M_W(S_t^x, W_{t+1})

**Why the distinction matters for VFA:**
- If you approximate V(S_t) (pre-decision), you need to take an expectation over W_{t+1} inside the optimization — expensive
- If you approximate V(S_t^x) (post-decision), the expectation is baked into the value function — the optimization becomes deterministic
- Using the wrong one leads to either: (a) incorrect Bellman updates, or (b) intractable expectations inside the policy

**Example:**
```
Pre-decision:   S_t = (inventory = 10, pending_orders = 5)
Decision:       x_t = order 20 more units
Post-decision:  S_t^x = (inventory = 10, pending_orders = 5, new_order = 20)
                 → deterministic given S_t and x_t
New info:       W_{t+1} = (demand = 8, delivery of previous order = 5)
Next state:     S_{t+1} = (inventory = 10 + 5 - 8 = 7, pending_orders = 20)
```

**Citation:** Powell, W.B. (2022). *RLSO*, Ch. 14-16 (pre-decision vs. post-decision state is central to ADP algorithms).

---

### Misunderstanding 7: Encoding Domain Knowledge as State Instead of Policy

**The mistake:** Adding "hint" variables to the state that encode what the *decision should be* rather than what the *situation is*.

**Example:**
```
BAD:   S_t = (inventory, demand_forecast, should_reorder_flag)
       → The "should_reorder_flag" is a policy output, not a state variable
       → It contaminates the state with a specific policy's logic

GOOD:  S_t = (inventory, demand_forecast)
       → Let the POLICY decide whether to reorder, based on the state
```

**Why it's damaging:**
- Bakes one policy's logic into the model, making it impossible to compare alternative policies fairly
- Violates the "model first, then solve" principle — the model should be policy-agnostic
- Leads to circular definitions: the state depends on the policy which depends on the state

**The fix:** State variables should describe the *situation*, never the *recommendation*. Policy-specific computed quantities belong in the policy, not the model.

---

### Misunderstanding 8: Ignoring Information Timing

**The mistake:** Including information in S_t that wouldn't actually be available at decision time t.

**Why it's critical:**
- S_t must only contain information known at time t, BEFORE the decision x_t
- Including future information (even partially) creates **look-ahead bias** — the policy appears to perform well in simulation but fails in deployment
- This is the sequential decision analog of **data leakage** in machine learning

**Common violations:**
- Using end-of-day price to make a beginning-of-day trading decision
- Using actual demand to make an ordering decision (instead of forecast/belief)
- Including outcomes of the current decision in the state used to make that decision

**The fix:** For every state variable, ask: "Is this known BEFORE I make decision x_t?" If not, it's either exogenous information W_{t+1} or it belongs in a future state.

---

### Summary: The State Design Checklist

For every candidate state variable, verify:

- [ ] **Needed for decisions?** Does the policy ever use this to choose x_t?
- [ ] **Needed for transitions?** Is it required to compute S_{t+1}?
- [ ] **Available at decision time?** Is it known before x_t is chosen?
- [ ] **Not derivable?** Can it be computed from other state variables? If so, remove it
- [ ] **Belief states included?** If learning about unknowns, are posterior parameters in S_t?
- [ ] **Policy-agnostic?** Does it describe the situation, not the recommendation?
- [ ] **Minimal?** Would removing it change the optimal decision? If not, remove it
- [ ] **Markov?** Given S_t, is the future independent of history?

### Citations
- Powell, W.B. (2022). *RLSO*, Ch. 11 (State Variables — the most detailed treatment).
- Powell, W.B. (2022). *SDAM*, Ch. 1 and Ch. 7 (State variables with examples).
- Powell, W.B. (2020). "On State Variables, Bandit Problems and POMDPs."
