# Skill: Sequential Decision Analytics — Problem Framing

## 1. Purpose

This skill enables the systematic translation of any real-world problem into the Sequential Decision Analytics (SDA) framework developed by Warren B. Powell. The central philosophy is **"Model first, then solve"** — an approach that is standard practice in deterministic optimization (linear programming, integer programming, network optimization) but is rarely applied to problems involving uncertainty. In stochastic domains, practitioners routinely jump straight to a solution method (reinforcement learning, dynamic programming, heuristic rules) without first writing down a proper model. This leads to poor problem understanding, missed structure, and suboptimal solutions.

This skill teaches you to **think like a sequential decision analyst**. Before writing a single line of algorithm code, you will:

- Precisely define what is being decided, when, and by whom.
- Identify what is uncertain and how information is revealed over time.
- Write down the five core elements of any sequential decision problem.
- Classify the problem to guide policy design.
- Only then select and implement a solution approach.

The SDA framework unifies fields that have historically operated in isolation — stochastic control, reinforcement learning, Markov decision processes, stochastic programming, bandit problems, simulation optimization, and approximate dynamic programming — under a single coherent modeling language. By learning to frame problems in this language, you gain access to the full spectrum of solution strategies rather than being locked into the conventions of one subfield.

**Key references:** Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions*. Wiley. Powell, W.B. (2011). *Approximate Dynamic Programming: Solving the Curses of Dimensionality*, 2nd Edition. Wiley.

---

## 2. Recognizing a Sequential Decision Problem

A problem is a **sequential decision problem** if it involves:

- **A sequence of decisions made over time** (or over iterations, trials, experiments, stages).
- **Uncertainty that is revealed between decisions** — you learn something new after each action.
- **The need to balance immediate and future consequences** — what you do now affects what happens later.
- The fundamental pattern: **Decision -> Information -> Decision -> Information -> ...**

Almost every interesting real-world problem has this structure. If you see any of the following, you are looking at a sequential decision problem:

- An agent choosing actions over time under uncertainty.
- A resource being allocated, consumed, or repositioned across periods.
- An experiment or trial where outcomes inform future choices.
- A system being controlled where disturbances arrive unpredictably.
- A process where you must commit to a choice before knowing the full outcome.

### Diagnostic Questions

Ask these questions to confirm you have a sequential decision problem and to begin scoping it:

1. **What decisions are being made?** List every choice, large or small.
2. **Who or what is making the decisions?** A person, a committee, an algorithm, a robot?
3. **When are decisions made?** Every minute? Every day? At irregular events? Once per iteration?
4. **What information arrives between decisions?** What do you learn after acting?
5. **How do decisions affect future states?** Does choosing action A now change what is possible or desirable later?
6. **What is the objective?** Minimize cost? Maximize reward? Maximize probability of success? Minimize risk? A combination?
7. **What are the constraints on decisions?** Budget limits, capacity, feasibility requirements, regulatory rules?
8. **Is this a finite or infinite horizon problem?** Is there a known endpoint, or does the process continue indefinitely?
9. **Is the agent learning about the environment, or managing known resources, or both?** This is critical for classification.
10. **What would a naive or current policy look like?** Understanding the baseline helps you see what the sequential decision framework can improve.

If the answer to questions 1-5 all indicate sequential structure with uncertainty, you have an SDA problem. Proceed to modeling.

---

## 3. The Three Phases of Modeling

### Phase I: Problem Framing

Problem framing is the most important and most neglected step. The goal is to translate a vague real-world description into a precise structural understanding. Work through these steps in order:

#### Step 1: Identify the Decision-Maker(s)

- Who or what is making sequential decisions?
- Is there a single decision-maker or multiple? (If multiple, is it cooperative or competitive?)
- For this framework, focus on a **single agent** making decisions. Multi-agent problems can often be decomposed.
- Examples: a warehouse manager, a trading algorithm, a clinical trial designer, an energy grid controller, a ride-sharing platform dispatcher.

#### Step 2: Identify the Temporal Structure

- What is the time scale? Seconds, minutes, hours, days, weeks, months, years?
- Is time **discrete** (decisions at fixed intervals) or **continuous** (decisions triggered by events)?
- For modeling, we typically discretize into periods t = 0, 1, 2, ..., T (or t = 0, 1, 2, ... for infinite horizon).
- What is the planning horizon? Finite (T is known) or infinite (ongoing)?
- How many time steps are there realistically? Tens? Hundreds? Thousands? This affects computational feasibility.

#### Step 3: List All Decisions

- Enumerate every action the decision-maker can take at each time step.
- Be specific: not "manage inventory" but "choose order quantity q_t >= 0 for each product."
- Characterize the decision space:
  - Binary (yes/no, accept/reject)
  - Integer (how many units)
  - Continuous (what price, what allocation fraction)
  - Categorical (which option from a finite set)
  - Vector-valued (multiple simultaneous choices)
- Identify constraints: what makes a decision feasible or infeasible?

#### Step 4: Identify Uncertainties

- What is unknown at the time each decision is made?
- What new information arrives after a decision and before the next one?
- Categories of uncertainty:
  - **Outcome uncertainty:** the result of our action is random (drug efficacy, investment return).
  - **Exogenous uncertainty:** external events happen regardless of our action (weather, demand, prices).
  - **Model uncertainty:** we do not know the parameters of the system (unknown mean, unknown transition probabilities).
- How is uncertainty resolved? All at once? Gradually? Through observation?

#### Step 5: Define the Goal

- What constitutes good performance?
- Is it: minimize total cost, maximize total reward, maximize probability of meeting a target, minimize worst-case loss, maximize information gained?
- Is there a discount factor (future rewards worth less than immediate ones)?
- Are there multiple competing objectives?
- Is risk-sensitivity important? (Variance, CVaR, worst-case considerations)

#### Step 6: Classify the Problem Type

This classification guides which solution strategies are most promising:

| Problem Type | Key Feature | Examples |
|---|---|---|
| **Pure learning** | Decisions exist only to gather information; no physical resources to manage | Multi-armed bandits, A/B testing, clinical trials, hyperparameter tuning, materials discovery |
| **Dynamic resource allocation** | Managing tangible resources under uncertainty; model parameters are known | Fleet management, inventory control, nurse scheduling, energy storage, portfolio rebalancing |
| **State-dependent control** | Controlling a physical system; state evolves dynamically | Robotics, autonomous vehicles, HVAC control, process control |
| **Hybrid learning + resource allocation** | Must simultaneously learn unknown parameters AND manage resources | New product pricing, adaptive clinical trials with resource constraints, online advertising with unknown click rates |

Most real problems are **hybrid**. Recognizing the learning component prevents you from ignoring the belief state, and recognizing the resource component prevents you from treating it as a pure bandit.

---

### Phase II: Mathematical Modeling — The Five Core Elements

Every sequential decision problem, regardless of domain, is completely characterized by five elements. Defining these five elements IS the model. Everything else — algorithms, policies, computation — comes after.

#### Element 1: State Variables S_t

The state S_t is everything you need to know at time t to make a decision and model forward. It is the complete "snapshot" of the system. The state has up to three components:

**Physical/Resource State R_t:**
- Tangible, countable, or measurable quantities.
- Examples: inventory levels by product, number of vehicles at each location, energy stored in a battery, cash in an account, position and velocity of a robot, number of nurses currently on shift.
- Ask: "What physical quantities change over time due to decisions and random events?"

**Information/Belief State B_t:**
- What you believe about unknown quantities, represented as probability distributions or sufficient statistics.
- Examples: estimated mean demand (plus confidence), posterior distribution over drug efficacy, learned value function parameters, Bayesian prior over customer preferences.
- Ask: "What am I uncertain about that I am learning over time? How do I represent my current beliefs?"
- For conjugate Bayesian models, the belief state can be represented compactly by sufficient statistics (e.g., for a normal distribution with unknown mean: the current estimate of the mean and the precision).
- **Common mistake:** Forgetting the belief state entirely. If your problem has any learning component, B_t must be part of S_t.

**Other Information I_t:**
- Any other relevant information that is neither physical resource nor belief.
- Examples: time of day, day of week, weather forecast, current market price, competitor's last action, a customer's stated preferences, lagged exogenous variables.
- Ask: "What other context affects the decision or the transition?"

**State Completeness Checks:**
- **Markov check:** Given S_t alone (without knowing S_{t-1}, S_{t-2}, ...), can you fully determine the probability distribution over S_{t+1} for any decision x_t? If not, you are missing state variables. Add whatever lagged information is needed until the state is Markov.
- **Decision sufficiency check:** Given S_t alone, do you have everything you need to evaluate the quality of any feasible decision x_t? If not, you are missing state variables.
- **Parsimony check:** Is every component of S_t actually used in either the decision, the transition, or the objective? Remove anything that is not.

#### Element 2: Decision Variables x_t

The decision (also called action or control) x_t is what the agent chooses at time t.

**Specification checklist:**
- What are the decision variables? (Scalar, vector, matrix?)
- What is the type of each variable? (Binary, integer, continuous, categorical?)
- What is the feasible set X_t(S_t)? How does feasibility depend on the current state?
  - Budget constraints: sum of allocations <= available budget.
  - Capacity constraints: cannot ship more than is in inventory.
  - Physical constraints: robot speed limits, battery charge rate limits.
  - Logical constraints: binary exclusion, precedence requirements.
- Is the decision made before or after observing W_t? (In our convention, x_t is chosen knowing S_t but before W_{t+1} is revealed.)

**Important distinction:** The decision x_t is a single instance. A **policy** is a function pi that maps states to decisions: x_t = X^pi(S_t). We optimize over policies, not individual decisions.

#### Element 3: Exogenous Information W_{t+1}

The exogenous information is everything that changes between time t and t+1 that is NOT caused by our decision x_t. It represents the arrival of new information and random events.

**Specification checklist:**
- What random events occur? List them all.
- What is the source of randomness? Nature, other agents, measurement noise?
- How is W_{t+1} distributed? Known distribution? Historical data? Scenario-based?
- Does W_{t+1} depend on the current state S_t? (It often does — e.g., demand depends on season, which is part of the state.)
- Does W_{t+1} depend on our decision x_t? (Be careful: if the outcome distribution depends on x_t, this is important to model correctly. For example, in a clinical trial, the observation depends on which treatment arm was chosen.)
- Modeling approaches:
  - **Parametric distributions:** W ~ N(mu, sigma^2), Poisson(lambda), etc.
  - **Historical data streams:** Replay actual historical data.
  - **Scenario generation:** Monte Carlo sampling from a model.
  - **Adversarial/robust:** Worst-case W from an uncertainty set.

#### Element 4: Transition Function S_{t+1} = S^M(S_t, x_t, W_{t+1})

The transition function (also called the system model or dynamics) describes how the state evolves. Given the current state, the decision, and the exogenous information, the next state is **deterministically** computed.

**Specification checklist:**

*Physical state transition R_{t+1} = f^R(R_t, x_t, W_{t+1}):*
- How do physical quantities change? Write the equations.
- Examples:
  - Inventory: R_{t+1} = R_t + x_t (order) - D_{t+1} (demand from W_{t+1}), subject to R_{t+1} >= 0.
  - Battery: R_{t+1} = R_t + eta_charge * x_t^charge - (1/eta_discharge) * x_t^discharge - loss.
  - Cash: R_{t+1} = R_t - cost(x_t) + revenue(W_{t+1}).

*Belief state transition B_{t+1} = f^B(B_t, x_t, W_{t+1}):*
- How do beliefs update given new observations?
- **Bayesian updating** is the gold standard:
  - Prior B_t + likelihood of observation -> Posterior B_{t+1}.
  - For normal-normal: update mean and precision using conjugate formulas.
  - For beta-Bernoulli: update alpha and beta counts.
- Frequentist alternatives: recursive least squares, exponential smoothing, Kalman filtering.
- The belief update often depends on x_t because the decision determines what we observe (e.g., we only learn about the drug we tested).

*Other information transition I_{t+1} = f^I(I_t, W_{t+1}):*
- How does contextual information evolve? (Often exogenously driven.)
- Examples: clock advances, new forecast arrives, prices update.

**Determinism check:** Given specific values of (S_t, x_t, W_{t+1}), is S_{t+1} uniquely determined? It must be. All randomness lives in W_{t+1}, not in the transition function itself.

#### Element 5: Objective Function

The objective function defines what we are optimizing. In SDA, **we optimize over policies**, not over individual decisions.

**Standard forms:**

*Expected cumulative reward/cost (most common):*
```
max_pi  E [ sum_{t=0}^{T} gamma^t * C(S_t, x_t^pi) ]
```
or
```
min_pi  E [ sum_{t=0}^{T} gamma^t * C(S_t, x_t^pi) ]
```
where:
- pi is the policy (a function from states to decisions).
- x_t^pi = X^pi(S_t) is the decision prescribed by policy pi in state S_t.
- C(S_t, x_t) is the single-period contribution (reward or cost).
- gamma in (0, 1] is the discount factor (gamma = 1 for undiscounted finite horizon).
- E is the expectation over all future exogenous information.
- T is the horizon (T = infinity for infinite horizon problems).

*Terminal reward:*
```
max_pi  E [ C_T(S_T) ]
```
Relevant when only the final state matters (e.g., maximize terminal wealth).

*Risk-adjusted:*
```
max_pi  E[C] - lambda * Var[C]
```
or using CVaR, or other risk measures.

*Constraints in expectation:*
```
max_pi  E[C]  subject to  Pr(S_t in safe set) >= 1 - epsilon  for all t
```

**Key insight:** The expectation is taken over the entire sequence of future random events. We cannot evaluate a policy by looking at a single scenario; we need the expected performance across many possible futures. In practice, this means **simulation** is almost always part of evaluation.

---

### Phase III: Engineering — Designing and Evaluating Policies

Once the model is fully specified (all five elements defined), you move to engineering: choosing, implementing, and evaluating policies. This skill focuses on framing, but the connection to policy design is essential.

**Steps:**

1. **Choose candidate policy classes.** Powell identifies four meta-classes:
   - **PFA (Policy Function Approximation):** Parameterized rules — lookup tables, if-then rules, analytic functions. Examples: (s, S) inventory policy, linear decision rules, threshold policies.
   - **CFA (Cost Function Approximation):** Modify a deterministic optimization to handle uncertainty — add safety stocks, penalties, robust constraints. Solve a (modified) optimization at each step.
   - **VFA (Value Function Approximation):** Approximate the value of being in a state and choose actions that maximize immediate reward plus estimated future value. Includes Q-learning, approximate DP.
   - **DLA (Direct Lookahead Approximation):** Explicitly model the future — solve a stochastic or deterministic lookahead problem at each decision point. Includes stochastic programming, model predictive control, rollout policies, Monte Carlo tree search.

2. **Select the computational approach:**
   - Simulation-based evaluation (almost always).
   - Analytical solutions (rarely available, but check — some inventory and bandit problems have known optimal policies).
   - Hybrid approaches (e.g., simulate outer loop, optimize inner loop).

3. **Implement, test, and iterate:**
   - Start with a simple baseline policy (e.g., a myopic or rule-based PFA).
   - Progressively try more sophisticated policies.
   - Compare policies on the SAME set of sample paths (common random numbers) for fair evaluation.
   - Report not just means but confidence intervals, distributions, and worst-case performance.

---

## 4. Problem Classification Taxonomy

Classifying your problem along multiple dimensions helps you identify the right modeling choices and solution strategies.

### By Information Structure

| Classification | Description | Implications |
|---|---|---|
| **Full observability** | The entire state S_t is directly observed | Standard MDP formulation; no need for belief tracking |
| **Partial observability** | Some components of the state are hidden or noisy | POMDP formulation; must maintain a belief state |
| **Known model** | Transition probabilities and distributions are known | Pure exploitation; no need for learning |
| **Unknown model** | Model parameters must be learned from data | Exploration-exploitation trade-off; belief state needed |
| **Stationary** | The underlying system does not change over time | Can learn a fixed policy; convergence guarantees possible |
| **Non-stationary** | The system changes over time (concept drift, regime changes) | Must continuously adapt; discounting or windowing old data |

### By Decision Structure

| Classification | Description | Examples |
|---|---|---|
| **Scalar decisions** | Single numeric choice | Order quantity, price to set, dose to administer |
| **Vector decisions** | Multiple simultaneous choices | Resource allocation across categories, portfolio weights |
| **Binary decisions** | Yes/no, stop/continue | Accept/reject an offer, stop a clinical trial, approve a loan |
| **Sequential selection** | Choose items one at a time from a set | Job scheduling, task assignment, route construction |
| **Hierarchical decisions** | High-level and low-level decisions interact | Strategic vs. tactical vs. operational planning |

### By Time Structure

| Classification | Description | Examples |
|---|---|---|
| **Finite horizon** | Known endpoint T | Project planning, clinical trial (fixed number of patients), seasonal inventory |
| **Infinite horizon** | No endpoint; ongoing process | Perpetual inventory, long-run asset management, ongoing service operations |
| **Event-driven** | Decisions triggered by events, not a clock | Customer arrivals, equipment failures, market orders |
| **Variable-length episodes** | Horizon depends on outcomes | Game playing, search and rescue, disease treatment |

### By Learning Structure

| Classification | Description | Policy Implications |
|---|---|---|
| **Pure exploitation** | All parameters known; no learning needed | Use PFA, CFA, or DLA with known model |
| **Pure exploration** | Goal is to maximize information, not immediate reward | Design of experiments, scientific exploration |
| **Exploration-exploitation trade-off** | Must balance learning with earning | Bandit algorithms, knowledge gradient, Thompson sampling, UCB |
| **Offline learning + online execution** | Learn from historical data, deploy fixed policy | Batch RL, supervised learning of policies |
| **Online learning** | Learn and act simultaneously in real time | Contextual bandits, online RL, adaptive control |

### By State/Decision Scale

| Classification | State Space | Decision Space | Typical Approaches |
|---|---|---|---|
| **Small-small** | Discrete, < 10^4 states | Discrete, < 10^2 actions | Exact DP, tabular methods |
| **Large-small** | Large or continuous state space | Small discrete action set | VFA, DQN, policy gradient |
| **Small-large** | Small state space | Large or continuous action set | CFA, mathematical programming |
| **Large-large** | Large/continuous states and actions | Large/continuous actions | CFA, DLA, actor-critic, parameterized PFA |

---

## 5. Worked Example: Framing a New Problem

### The Vague Problem Statement

> "A hospital needs to decide how many nurses to schedule each shift, given uncertain patient arrivals, while minimizing staffing costs and maintaining quality of care."

### Phase I: Problem Framing

**Step 1 — Decision-maker:** The hospital's staffing coordinator (or an algorithm advising them). Single agent.

**Step 2 — Temporal structure:** Decisions are made once per shift (e.g., 8-hour shifts, so 3 decisions per day). Discrete time: t = 0, 1, 2, ..., T where each t is one shift. Horizon could be finite (plan the next 4 weeks, T = 84) or rolling/infinite.

**Step 3 — Decisions:**
- x_t = number of nurses to schedule for shift t (integer, x_t >= 0).
- Possible additional decisions: whether to call in on-call nurses mid-shift (a recourse decision); whether to offer overtime to currently working nurses.
- For simplicity, start with: x_t = number of nurses scheduled at the start of shift t.
- Constraints: x_t >= x_min (minimum safe staffing), x_t <= x_max (total nurse pool), labor contract constraints (cannot exceed weekly hours per nurse).

**Step 4 — Uncertainties:**
- Patient arrivals during each shift: random, possibly dependent on time of day, day of week, season, epidemics.
- Patient acuity (how much nursing time each patient requires): random.
- Nurse absenteeism: some scheduled nurses may not show up.
- Possible model uncertainty: if the hospital is new or patient patterns are changing, the arrival rate itself is unknown and must be learned.

**Step 5 — Goal:**
- Minimize expected total staffing cost over the planning horizon.
- Subject to quality constraints: probability that patient wait time exceeds threshold must be below some limit; nurse-to-patient ratio must remain above minimum.
- Possible multi-objective: minimize cost AND maximize quality score.
- For now, model as: minimize E[sum_t (wage_cost(x_t) + overtime_cost(x_t) + penalty_for_understaffing(S_t, x_t, W_{t+1}))].

**Step 6 — Classification:**
- Primarily a **dynamic resource allocation** problem (scheduling nurses = allocating a resource over time).
- Has a **learning component** if patient arrival rates are uncertain and being estimated.
- If arrival patterns are well-known (stable hospital, years of data): pure resource allocation.
- If arrival patterns are changing or poorly known: hybrid learning + resource allocation.
- Time structure: finite horizon (4-week schedule) with rolling replanning.

### Phase II: Mathematical Modeling — Five Elements

**State Variables S_t = (R_t, B_t, I_t):**

- Physical state R_t:
  - N_t^{on} = number of nurses currently on duty (from the previous shift, if shifts overlap).
  - N_t^{avail} = number of nurses available for scheduling (total pool minus those on mandatory rest, leave, etc.).
  - P_t = current patient census (number of patients in the hospital at the start of shift t).
  - H_t = vector of cumulative hours worked this week for each nurse (for labor constraint tracking).

- Belief state B_t (if learning):
  - Estimated patient arrival rate: mu_t^{hat} (e.g., the current estimate of average arrivals per shift).
  - Precision of estimate: n_t (number of observations incorporated; higher means more confident).
  - Could be represented as a normal-gamma posterior if using Bayesian updating on arrival counts.

- Other information I_t:
  - Shift type: morning, afternoon, night (affects arrival patterns and costs).
  - Day of week and whether it is a holiday.
  - Season or flu-season indicator.
  - Any known upcoming events (e.g., scheduled surgeries, expected transfers).

- **Markov check:** Given (N_t^{on}, N_t^{avail}, P_t, H_t, mu_t^{hat}, n_t, shift_type, day, season), can we model the next state without needing earlier history? Yes — if we have patient census and cumulative hours, we do not need the full history. The state is Markov.

**Decision Variables x_t:**

- x_t = number of nurses to schedule for shift t. Integer, x_t >= 0.
- Feasibility: x_t <= N_t^{avail}; x_t >= x_min(shift_type); for each nurse, weekly hours including this shift must not exceed contract maximum.
- For a richer model: x_t could be a vector indicating which specific nurses are assigned, enabling skill-matching and preference-based scheduling.

**Exogenous Information W_{t+1}:**

- D_{t+1} = number of patient arrivals during shift t (realized after staffing decision x_t is made).
- A_{t+1} = vector of patient acuity scores for arriving patients.
- Z_{t+1} = number of scheduled nurses who call in sick (nurse absenteeism).
- Q_{t+1} = number of patients discharged during the shift.
- If the arrival rate is unknown and being learned, we observe D_{t+1} which updates our belief.

**Transition Function S_{t+1} = S^M(S_t, x_t, W_{t+1}):**

- Nurses on duty: N_{t+1}^{on} = x_t - Z_{t+1} (scheduled minus absent, though outgoing nurses from the previous shift leave).
- Available pool: N_{t+1}^{avail} = updated based on rest requirements and who just finished shifts.
- Patient census: P_{t+1} = P_t + D_{t+1} - Q_{t+1} (arrivals minus discharges).
- Hours tracking: H_{t+1,i} = H_{t,i} + 8 if nurse i was scheduled, with weekly reset.
- Belief update (if learning): observe D_{t+1}; update mu_t^{hat} and n_t using Bayesian updating:
  - n_{t+1} = n_t + 1
  - mu_{t+1}^{hat} = (n_t * mu_t^{hat} + D_{t+1}) / n_{t+1}

- **Determinism check:** Given specific values of (S_t, x_t, D_{t+1}, A_{t+1}, Z_{t+1}, Q_{t+1}), the next state is uniquely determined. Confirmed.

**Objective Function:**

```
min_pi  E [ sum_{t=0}^{T} ( c_wage * x_t + c_overtime * max(0, x_t - x_regular)
          + c_under * max(0, P_t / (x_t - Z_{t+1}) - ratio_max)
          + c_wait * expected_wait_penalty(P_t, D_{t+1}, x_t - Z_{t+1}) ) ]
```

where x_t = X^pi(S_t) is determined by the policy. The first term is base staffing cost, the second is overtime cost, the third penalizes unsafe nurse-to-patient ratios, and the fourth penalizes excessive patient wait times.

### Phase III: Engineering — Candidate Policies

**PFA candidates:**
- Fixed staffing rule: x_t = f(shift_type, day_of_week) — a lookup table.
- Parameterized rule: x_t = ceil(alpha * P_t + beta * mu_t^{hat} + gamma), where alpha, beta, gamma are tuned by simulation.

**CFA candidates:**
- Solve a deterministic scheduling optimization each period using point forecasts, but inflate forecasted demand by a safety factor (the safety factor is the tunable parameter).
- Stochastic programming: two-stage model where first stage is scheduling, second stage is recourse (call in extra nurses) after observing arrivals.

**VFA candidates:**
- Approximate the value function V(S_t) and solve x_t = argmin [C(S_t, x_t) + gamma * E[V(S^M(S_t, x_t, W_{t+1}))]].
- Represent V using a linear architecture: V(S_t) = theta_0 + theta_1 * P_t + theta_2 * N_t^{avail} + ...
- Train using approximate dynamic programming (backward or forward).

**DLA candidates:**
- Rolling-horizon deterministic lookahead: at each shift, solve a deterministic scheduling problem for the next K shifts using forecasted demand.
- Monte Carlo tree search: simulate multiple demand scenarios for the next K shifts and optimize over the tree.

**Recommended starting point:** Begin with the parameterized PFA, as it is simple to implement and fast to evaluate. Use simulation over many sample paths to tune the parameters. Then compare against the CFA approach (deterministic optimization with safety factors). If neither performs well enough, invest in VFA or DLA.

---

## 6. Framing Checklist (Quick Reference)

Use this checklist to rapidly frame any sequential decision problem. For each item, write down the answer explicitly before moving on.

### Identity
- [ ] Who is the decision-maker?
- [ ] What is the time scale and horizon?
- [ ] Is the problem finite or infinite horizon?

### Decisions
- [ ] What actions can be taken at each time step?
- [ ] What type are the decision variables? (binary, integer, continuous, categorical, vector)
- [ ] What constraints restrict the decisions? How do they depend on the state?

### Uncertainty
- [ ] What is unknown at each decision point?
- [ ] What new information arrives after each decision?
- [ ] Is there model uncertainty (unknown parameters) that we are learning about?
- [ ] How is uncertainty modeled? (distributions, data, scenarios)

### State
- [ ] Physical/resource state R_t: What tangible quantities must be tracked?
- [ ] Belief state B_t: What beliefs about unknowns must be tracked?
- [ ] Other information I_t: What context or exogenous variables matter?
- [ ] Is the state Markov? If not, what is missing?
- [ ] Is every state component necessary? (no redundancy)

### Dynamics
- [ ] How does R_t transition? Write the equation.
- [ ] How does B_t update? (Bayesian updating formula, or alternative)
- [ ] How does I_t evolve?
- [ ] Is the transition deterministic given (S_t, x_t, W_{t+1})?

### Objective
- [ ] Are we minimizing or maximizing?
- [ ] What is the single-period contribution C(S_t, x_t)?
- [ ] Is there discounting? What is gamma?
- [ ] Are there risk considerations?
- [ ] Are we optimizing over policies (not just single decisions)?

### Classification
- [ ] Problem type: pure learning / resource allocation / control / hybrid?
- [ ] Information structure: full/partial observability? known/unknown model?
- [ ] Decision structure: scalar / vector / binary / sequential?
- [ ] Time structure: finite / infinite / event-driven?
- [ ] Learning structure: exploitation / exploration / trade-off?

### Policy Design
- [ ] What is the simplest baseline policy? (This is your starting point.)
- [ ] Which of the four meta-classes (PFA, CFA, VFA, DLA) are natural candidates?
- [ ] What computational resources are available?
- [ ] How will policies be evaluated? (simulation, analytical, field testing)

---

## 7. Common Patterns

When you recognize a problem pattern, you can immediately identify strong candidate solution approaches:

### Inventory and Ordering Problems
- **Pattern:** Decide how much to order to replenish stock, given random demand.
- **State:** Inventory level R_t, possibly belief about demand distribution.
- **Classic policies:** (s, S) policy (order up to S when inventory drops below s), (r, q) policy (order q when inventory reaches reorder point r). These are PFAs with tunable parameters.
- **Recommended approach:** Start with parameterized PFA; use simulation optimization to tune (s, S) or (r, q) parameters. For multi-product with interactions, use CFA (linear program with safety stocks).

### Routing and Scheduling Problems
- **Pattern:** Assign resources (vehicles, workers, machines) to tasks over time.
- **State:** Resource locations/availability, pending tasks, time.
- **Decisions:** Which resource handles which task; in what sequence.
- **Recommended approach:** Deterministic DLA (solve a deterministic assignment/routing problem at each stage using current information) or CFA (myopic optimization with penalties for leaving tasks unserved). For large scale, use PFA with priority rules.

### Learning and Experimentation Problems
- **Pattern:** Choose which option to test/explore to learn about unknown qualities.
- **State:** Belief state B_t (posterior distributions over option qualities).
- **Decisions:** Which option to try next.
- **Key challenge:** Exploration vs. exploitation trade-off.
- **Recommended approach:** Knowledge gradient (a VFA-based approach that values the information gained by each choice), Thompson sampling (a PFA that samples from the posterior), upper confidence bound (UCB, a PFA using optimistic estimates).

### Resource Allocation Under Uncertainty
- **Pattern:** Allocate limited resources across competing uses, with uncertain outcomes or demands.
- **State:** Resource levels, demand forecasts, possibly beliefs about return rates.
- **Decisions:** How much to allocate to each use.
- **Recommended approach:** CFA (solve a resource allocation LP/QP at each step with modified parameters for uncertainty). If the state space is manageable, VFA with a convex approximation architecture.

### Real-Time Control Problems
- **Pattern:** Control a physical system (temperature, position, chemical process) with disturbances.
- **State:** Physical system state (position, velocity, temperature, concentrations).
- **Decisions:** Control inputs (forces, valve settings, heating/cooling rates).
- **Recommended approach:** PFA (PID controllers, linear feedback), CFA (model predictive control — solve a deterministic or stochastic lookahead optimization at each step), or DLA with short horizon.

### Online Pricing and Revenue Management
- **Pattern:** Set prices over time to maximize revenue given uncertain demand and limited inventory.
- **State:** Remaining inventory, time until deadline, belief about price-demand relationship.
- **Decisions:** Price to charge in each period.
- **Recommended approach:** If demand model is known, use VFA or DLA. If demand model is unknown, hybrid approach: Thompson sampling for learning demand + VFA or CFA for pricing given current beliefs.

### Sequential Testing and Diagnosis
- **Pattern:** Choose which test to perform next to identify a condition, minimizing cost or time.
- **State:** Belief about possible conditions (posterior probabilities), test results so far.
- **Decisions:** Which test to perform next, or when to stop and declare a diagnosis.
- **Recommended approach:** VFA (value of information calculations) or DLA (lookahead over possible test sequences).

---

## 8. Anti-Patterns and Pitfalls

### Anti-Pattern 1: Jumping to a Solution Method Before Modeling

**Symptom:** "We have a scheduling problem, so let's use reinforcement learning." No model is written down. State, decisions, transitions, and objective are defined implicitly by the code rather than explicitly in a model.

**Why it is harmful:** Without an explicit model, you cannot verify correctness, compare approaches fairly, or reason about what information matters. You may solve the wrong problem without realizing it.

**Fix:** Always complete Phase I and Phase II before writing algorithm code. Write down all five elements. Only then select a solution approach.

### Anti-Pattern 2: Defaulting to RL/DP Without Considering Simpler Alternatives

**Symptom:** Every problem is framed as an MDP and attacked with deep RL, even when a simple PFA or CFA would work well.

**Why it is harmful:** RL methods require extensive tuning, large amounts of data or simulation time, and often converge to policies that are hard to interpret. A well-tuned (s, S) policy or model predictive control approach may outperform deep RL while being simpler, faster, and more trustworthy.

**Fix:** Start with the simplest policy class that might work (usually PFA or CFA). Only escalate to VFA or DLA when simpler approaches demonstrably fail. Always benchmark against simple baselines.

### Anti-Pattern 3: Ignoring the Belief State

**Symptom:** The problem involves learning about unknown parameters, but the state definition includes only physical quantities. The algorithm learns implicitly (e.g., through a replay buffer or neural network weights) but the belief state is not modeled as part of S_t.

**Why it is harmful:** Without an explicit belief state, you cannot reason about the value of information, you cannot design principled exploration strategies, and the state is non-Markov (violating the foundational assumption).

**Fix:** Whenever there are unknown parameters being learned, include sufficient statistics or a posterior distribution in B_t. This makes the state Markov and enables information-aware policies.

### Anti-Pattern 4: Over-Specifying the State (Curse of Dimensionality)

**Symptom:** The state includes every conceivable variable — every product's inventory at every location, detailed history of every customer interaction, full joint distribution over all unknown parameters.

**Why it is harmful:** High-dimensional states make tabular methods impossible and make function approximation difficult. Computation and memory explode. Policies become slow to evaluate.

**Fix:** Include only variables that actually affect the decision or the transition. Aggregate where possible (e.g., total inventory across locations, rather than per-location, if decisions are made at the aggregate level). Use feature engineering to create compact state representations. Test whether adding/removing a variable actually changes policy quality.

### Anti-Pattern 5: Under-Specifying the State (Non-Markov Behavior)

**Symptom:** The policy makes poor decisions because it lacks information about the past. For example, a scheduling policy that does not account for recent demand trends, or an inventory policy that does not know remaining budget for the year.

**Why it is harmful:** If the state is not Markov, the transition probabilities and optimal policy depend on history in ways the model cannot capture. This leads to suboptimal and inconsistent decisions.

**Fix:** When a policy's performance depends on history, add the relevant historical summary to the state. Common additions: cumulative quantities (total spent, total produced), moving averages, trend indicators, lagged observations.

### Anti-Pattern 6: Confusing the Policy with the Model

**Symptom:** The model definition includes the solution method — for example, "we use Q-learning, so the state is whatever the neural network takes as input."

**Why it is harmful:** The model should be algorithm-agnostic. The same model (same five elements) should be solvable by PFA, CFA, VFA, or DLA. Conflating model and algorithm restricts your options and obscures the problem structure.

**Fix:** Define the five elements without any reference to how you will solve the problem. Only in Phase III do you choose an algorithm.

### Anti-Pattern 7: Ignoring the Objective Function Specification

**Symptom:** The objective is vaguely stated ("maximize efficiency") without a precise mathematical form. Or the implemented reward signal does not match the true objective (reward hacking).

**Why it is harmful:** Without a precise objective, you cannot evaluate policies meaningfully. A policy optimized for the wrong objective will behave poorly on the real objective, sometimes catastrophically.

**Fix:** Write down the objective function explicitly: the mathematical expression being maximized or minimized, whether expectation is involved, the discount factor, any constraints. Validate that optimizing this objective would actually lead to desired real-world behavior.

### Anti-Pattern 8: Single-Scenario Evaluation

**Symptom:** A policy is tested on one scenario (or one seed, or one historical trajectory) and declared successful.

**Why it is harmful:** Sequential decision problems involve uncertainty. A policy that looks good on one scenario may perform terribly on others. You are optimizing expected performance, which requires evaluating across many scenarios.

**Fix:** Always evaluate policies over many Monte Carlo scenarios (at least hundreds, preferably thousands). Report means, standard deviations, confidence intervals, and tail behavior. Use common random numbers for fair comparisons between policies.

---

## 9. Citations and References

### Primary References

- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions*. John Wiley & Sons. — The definitive reference for the SDA framework. Covers all four policy classes, the universal modeling framework, and extensive examples across domains.

- Powell, W.B. (2011). *Approximate Dynamic Programming: Solving the Curses of Dimensionality*, 2nd Edition. John Wiley & Sons. — Comprehensive treatment of ADP methods including value function approximation, policy search, and applications to resource allocation.

### Key Papers by Powell and Collaborators

- Powell, W.B. (2014). "Clearing the Jungle of Stochastic Optimization." *Bridging Data and Decisions*, INFORMS TutORials in Operations Research, pp. 109-137. — Foundational paper classifying the four policy classes.

- Powell, W.B. and Meisel, S. (2016). "Tutorial on Stochastic Optimization in Energy — Part I: Modeling and Policies." *IEEE Transactions on Power Systems*, 31(2), 1459-1467.

- Powell, W.B. and Meisel, S. (2016). "Tutorial on Stochastic Optimization in Energy — Part II: An Energy Storage Illustration." *IEEE Transactions on Power Systems*, 31(2), 1468-1475.

- Frazier, P.I., Powell, W.B., and Dayanik, S. (2008). "A Knowledge-Gradient Policy for Sequential Information Collection." *SIAM Journal on Control and Optimization*, 47(5), 2410-2439. — The knowledge gradient method for optimal learning.

- Ryzhov, I.O., Powell, W.B., and Frazier, P.I. (2012). "The Knowledge Gradient Algorithm for a General Class of Online Learning Problems." *Operations Research*, 60(1), 180-195.

- Nascimento, J.M. and Powell, W.B. (2009). "An Optimal Approximate Dynamic Programming Algorithm for the Lagged Asset Acquisition Problem." *Mathematics of Operations Research*, 34(1), 210-237.

- Powell, W.B. (2019). "A Unified Framework for Stochastic Optimization." *European Journal of Operational Research*, 275(3), 795-821.

- Powell, W.B. (2020). "From Reinforcement Learning to Optimal Control: A Unified Framework for Sequential Decisions." In *Handbook of Reinforcement Learning and Control*, Springer.

### Background and Related References

- Puterman, M.L. (2005). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. John Wiley & Sons. — Classic reference on MDPs and dynamic programming.

- Bertsekas, D.P. (2012). *Dynamic Programming and Optimal Control*, Vols. I and II, 4th Edition. Athena Scientific. — Comprehensive treatment of DP theory.

- Sutton, R.S. and Barto, A.G. (2018). *Reinforcement Learning: An Introduction*, 2nd Edition. MIT Press. — Standard reference for RL methods.

- Birge, J.R. and Louveaux, F. (2011). *Introduction to Stochastic Programming*, 2nd Edition. Springer. — Stochastic programming methods, relevant to DLA approaches.

- Szepesvari, C. (2010). *Algorithms for Reinforcement Learning*. Morgan and Claypool. — Concise treatment of RL algorithms.

- Gittins, J., Glazebrook, K., and Weber, R. (2011). *Multi-Armed Bandit Allocation Indices*, 2nd Edition. John Wiley & Sons. — Theory of bandit problems and index policies.

### Domain-Specific Applications of the SDA Framework

- Simao, H.P., Day, J., George, A.P., Gifford, T., Nienow, J., and Powell, W.B. (2009). "An Approximate Dynamic Programming Algorithm for Large-Scale Fleet Management." *Transportation Science*, 43(2), 178-197. — Large-scale fleet management application.

- Powell, W.B., George, A., Simao, H., Scott, W., Lamont, A., and Stewart, J. (2012). "SMART: A Stochastic Multiscale Model for the Analysis of Energy Resources, Technology, and Policy." *INFORMS Journal on Computing*, 24(4), 665-682. — Energy systems application.

- Mes, M.R.K. and Powell, W.B. (2022). "Approximate Dynamic Programming and Reinforcement Learning." In *International Series in Operations Research and Management Science*. Springer.
