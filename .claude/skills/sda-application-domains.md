# SDA Application Domains and Case Studies

## Skill Purpose

This skill provides deep knowledge of the application domains and case studies from Warren B. Powell's Sequential Decision Analytics (SDA) framework, drawn primarily from *Sequential Decision Analytics and Modeling* (SDAM, 2022) and *Reinforcement Learning and Stochastic Optimization* (RLSO, 2022). Use this skill to:

- Identify which SDA application pattern fits a user's problem
- Teach SDA concepts through concrete, well-documented examples
- Adapt published case studies to new domains
- Select appropriate policy classes for a given application type
- Model any sequential decision problem using Powell's universal five-element framework: state (S_t), decision (x_t), exogenous information (W_{t+1}), transition function (S^M), and objective function

---

## 1. Overview

### 1.1 Scope

SDA spans virtually every human process involving sequential decisions under uncertainty. Powell's two books collectively cover over 100 applications across energy, health, transportation, finance, supply chain, e-commerce, laboratory sciences, and military operations. The framework's power lies in its universality: every application, no matter how different on the surface, is modeled with the same five structural elements.

### 1.2 Teach-by-Example Methodology

Powell uses a deliberate teach-by-example style. Each application chapter follows a consistent outline:

1. **Narrative** -- a plain-language description of the problem and why it matters
2. **Model** -- formalize using the five elements (state, decision, exogenous information, transition, objective)
3. **Uncertainty characterization** -- what is random, how it is revealed, and what distributional assumptions apply
4. **Policies** -- which of the four policy classes (PFA, CFA, VFA, DLA) apply, and why
5. **Evaluation** -- how to measure policy quality (simulation, bounds, benchmarks)
6. **Extensions** -- how the base model can be enriched for realism

This structure means that once you understand one application deeply, you can read any other application chapter fluently.

### 1.3 The Four Policy Classes (Quick Reference)

Every application below selects policies from these four meta-classes:

| Abbreviation | Full Name | Core Idea |
|---|---|---|
| **PFA** | Policy Function Approximation | Analytical function mapping state to action (lookup tables, parametric rules, neural networks) |
| **CFA** | Cost Function Approximation | Modify an optimization model's objective/constraints with tunable parameters |
| **VFA** | Value Function Approximation | Approximate the downstream value (Bellman's equation) and optimize current decision + future value |
| **DLA** | Direct Lookahead Approximation | Build and solve a (possibly approximate) model of the future at each decision epoch |

### 1.4 The Universal Model

For every application below, the model takes the form:

```
max_{pi} E[ sum_{t=0}^{T} C(S_t, X^pi(S_t)) | S_0 ]
```

Where:
- `S_t` = state at time t (physical state R_t, information state I_t, belief state B_t)
- `X^pi(S_t)` = decision function (policy pi maps state to action)
- `W_{t+1}` = new exogenous information arriving after decision
- `S_{t+1} = S^M(S_t, x_t, W_{t+1})` = transition function
- `C(S_t, x_t)` = contribution (reward/cost) of decision x_t in state S_t

---

## 2. Pure Learning Problems

Pure learning problems focus on reducing uncertainty about unknown quantities through sequential observations. The physical state is minimal or absent; the belief state dominates.

### 2.1 Asset Selling Problem (SDAM Ch. 2)

**Narrative.** You own a single asset (e.g., a house, a used car, a block of stock). Each period, a buyer arrives and offers a random price. You must decide immediately: accept the offer and sell (terminal), or reject it and wait for a future offer. There may be a finite deadline or a discount factor penalizing delay.

**State variables.**
- `p_t` = current price offer
- `t` = time period (or time remaining before deadline)
- `B_t` = beliefs about the price distribution (if learning from observed offers)

**Decision.** Binary: sell (`x_t = 1`) or hold (`x_t = 0`). Selling is terminal.

**Exogenous information.** `W_{t+1} = p_{t+1}`, the next price offer, drawn from distribution `F(p)` (possibly unknown and being learned).

**Transition.**
- If sell: process ends
- If hold: `S_{t+1} = (p_{t+1}, t+1, B_{t+1})` where `B_{t+1}` is the updated belief after observing `p_t`

**Policies illustrated.**
- **PFA (threshold policy):** Sell if `p_t > p*` for a tunable threshold `p*`. Elegant and often near-optimal. The threshold can depend on time remaining.
- **VFA:** Compute the expected value of waiting, `V_t = E[max(p_{t+1}, V_{t+1})]`, and sell if `p_t >= V_t`. This is exact dynamic programming for small problems.

**Key insight.** This is an *optimal stopping problem* -- the simplest non-trivial SDA model. It demonstrates that even a basic PFA (threshold rule) can perform well, while VFA gives the exact solution. It is the entry point for understanding when PFAs suffice and when VFA adds value.

**Extensions.**
- Time-varying or non-stationary price distributions
- Transaction costs (cost of holding, cost of selling)
- Partial selling (sell a fraction of the asset each period)
- Multiple assets with correlated prices
- Bayesian learning of the price distribution parameters

---

### 2.2 Adaptive Market Planning (SDAM Ch. 3)

**Narrative.** A firm is planning market entry or capital investment under uncertainty about market response (demand, competitor actions, regulatory changes). Decisions are sequential: invest in phases, observe market feedback, adjust plans.

**State variables.**
- Investment committed so far
- Market signals observed (demand data, competitor moves)
- Beliefs about market parameters (growth rate, price elasticity)

**Decision.** Investment level for the current phase; pricing or capacity decisions.

**Exogenous information.** Market response data (sales, competitor entry, regulation changes).

**Policies illustrated.**
- **CFA approach:** Formulate a planning model (e.g., NPV maximization) and parameterize it with adjustable terms -- a safety buffer for demand uncertainty, a risk premium on investment. Tune parameters via simulation.
- The CFA modifies the objective or constraints of a deterministic optimization model to implicitly handle uncertainty.

**Key insight.** Demonstrates how CFAs work in practice: you do not solve a stochastic model directly. Instead, you solve a parameterized deterministic model and tune the parameters against simulated stochastic outcomes. This is how most real industrial planning works (even if practitioners do not use the term CFA).

---

### 2.3 Learning the Best Diabetes Medication (SDAM Ch. 4)

**Narrative.** A doctor must prescribe one of `M` medications to a patient with type 2 diabetes. Different patients respond differently to each medication (varying efficacy, side effects). The doctor wants to learn which medication is best for this patient through sequential prescriptions and observed outcomes.

**State variables.**
- `B_t = (\mu_t^m, \sigma_t^m)_{m=1}^{M}` -- Bayesian posterior beliefs about the true efficacy of each medication `m`. Each medication has a prior mean `\mu_0^m` and prior variance `(\sigma_0^m)^2`, updated after each observation.
- `n_t^m` = number of times medication `m` has been prescribed (trial count)

**Decision.** `x_t \in \{1, 2, ..., M\}` -- which medication to prescribe next.

**Exogenous information.** `W_{t+1} = \hat{\mu}^{x_t}_{t+1}` -- the observed patient response (efficacy score), a noisy signal of the true efficacy of the chosen medication.

**Transition.** Bayesian update of beliefs:
- For the chosen medication `x_t`: update `(\mu_t^{x_t}, \sigma_t^{x_t})` using the new observation via conjugate normal-normal updating
- For all other medications: beliefs unchanged

**Policies illustrated.**
- **Thompson sampling (PFA):** Sample from each medication's posterior; prescribe the one with the highest sample. Automatically balances exploration and exploitation.
- **Upper confidence bounding / UCB (PFA):** Prescribe medication maximizing `\mu_t^m + \kappa \sigma_t^m` for tunable exploration parameter `\kappa`.
- **Knowledge gradient (PFA):** Prescribe the medication that maximizes the expected single-period improvement in the value of the best medication's mean. This is a one-step lookahead on learning value.
- **Interval estimation (PFA):** Prescribe the medication with the highest upper end of a confidence interval.
- **Pure exploitation (PFA):** Always prescribe the medication with the highest current mean `\mu_t^m`. No exploration -- often suboptimal.
- **Pure exploration:** Prescribe each medication equally to gather information. Ignores exploitation -- also often suboptimal.

All of these are PFAs because they are direct mappings from belief state to action, with no downstream optimization model.

**Key insight.** This is the *multi-armed bandit* problem, and it makes the belief state central to SDA. The state is not physical inventory or location -- it is what you *know*. The exploration-exploitation tradeoff is a core tension: prescribing the medication you currently think is best (exploitation) vs. trying others to improve your knowledge (exploration). The knowledge gradient is Powell's preferred approach because it directly values the information gained.

**Extensions.**
- **Contextual bandits:** Patient features (age, weight, genetics) affect response. The belief state becomes a function mapping contexts to efficacy.
- **Correlated beliefs:** Medications in the same drug class share information -- observing one updates beliefs about related ones.
- **Finite horizon with discounting:** The doctor has a limited number of visits, increasing the cost of exploration.
- **Multi-objective:** Balance efficacy against side effects, cost, and patient preference.

---

### 2.4 Stochastic Shortest Path -- Static (SDAM Ch. 5)

**Narrative.** A traveler must find the shortest path through a network (graph) where edge traversal costs are uncertain but *stationary* (they do not change over time, though the traveler does not know them). Each time the traveler traverses an edge, they observe its actual cost and update their beliefs.

**State variables.**
- `i_t` = current node in the network
- `B_t = (\mu_t^e, \sigma_t^e)$ for each edge `e`$ -- beliefs about the true cost of each edge

**Decision.** Which adjacent edge to traverse next.

**Exogenous information.** `W_{t+1} = c^{e_t}` -- the realized cost of the traversed edge (noisy observation of the true cost).

**Transition.**
- Move to the next node: `i_{t+1}` = destination of chosen edge
- Update beliefs for the traversed edge via Bayesian updating
- Other edges' beliefs unchanged (unless correlated)

**Policies illustrated.**
- **VFA:** Approximate the value (cost-to-go) of reaching each node, `\bar{V}(i)`, and choose the edge minimizing immediate cost plus downstream value. This is approximate dynamic programming on the network.
- **DLA:** Plan ahead by solving a deterministic shortest-path problem using current belief means, then follow the first edge of that plan. Re-plan after each step with updated beliefs.
- **PFA with exploration bonus:** Choose edges that balance low expected cost with high uncertainty (analogous to UCB on edges).

**Key insight.** Combines learning (about edge costs) with resource allocation (choosing a path). The traveler faces the exploration-exploitation dilemma spatially: should you take the route you believe is shortest, or detour to learn about edges that might be even shorter? This bridges the pure learning problems (Ch. 2-4) and the resource allocation problems (Ch. 8+).

---

### 2.5 Stochastic Shortest Path -- Dynamic (SDAM Ch. 6)

**Narrative.** Same network setting as 2.4, but now edge costs change over time (e.g., traffic congestion varies by hour, weather affects road conditions). The traveler must account for both uncertainty *and* non-stationarity.

**State variables.**
- `i_t` = current node
- `B_t` = beliefs about current and future edge cost distributions (may include time-of-day models, trend estimates)
- Possibly exogenous state variables (time of day, weather forecast)

**Decision.** Which adjacent edge to traverse next.

**Exogenous information.** `W_{t+1}` includes both the realized cost of the traversed edge and changes to other edges' costs (which may or may not be observed).

**Policies illustrated.**
- **DLA (dominant):** Solve a lookahead model that forecasts future edge costs and plans the best path over the forecast horizon. Re-solve at each step. Google Maps is a real-world DLA: it uses current traffic data to forecast future conditions and computes a deterministic shortest path over that forecast.
- **VFA:** Harder because the value function depends on time-varying exogenous state.
- **Hybrid DLA+learning:** Incorporate exploration bonuses into the lookahead model to value information gathering.

**Key insight.** Non-stationarity makes the problem fundamentally harder. Past observations become less informative as the environment changes. DLA naturally handles this because it re-plans at every step using the latest information. This chapter motivates DLA as the natural approach for dynamic, non-stationary environments.

**Real-world connection.** GPS navigation apps are DLAs operating in real time on dynamic stochastic networks.

---

## 3. Energy Applications

Energy is Powell's canonical demonstration domain for showing that all four policy classes can be optimal depending on problem structure.

### 3.1 Energy Storage I and II (SDAM Chs. 8-9)

**Narrative.** An operator manages a battery or energy storage system connected to a grid with uncertain prices and possibly uncertain renewable (wind/solar) supply. The goal is to maximize profit by buying energy when cheap, storing it, and selling when expensive. This is the *central teaching example* of SDAM because Powell designs five variants of this problem, each structured so a different policy class is optimal.

**State variables.**
- `R_t` = energy currently stored (physical state, continuous, bounded by capacity)
- `p_t` = current energy price (exogenous state)
- Possibly: `\hat{p}_t` = price forecast, `w_t` = wind/solar output, `B_t` = beliefs about price process

**Decision.** `x_t = (x_t^{buy}, x_t^{sell}, x_t^{store})` -- how much energy to buy from the grid, sell to the grid, and store. Subject to:
- Storage capacity: `0 \le R_t + x_t^{buy} - x_t^{sell} \le R^{max}`
- Non-negativity: `x_t^{buy}, x_t^{sell} \ge 0`
- Rate constraints: charge/discharge rate limits

**Exogenous information.** `W_{t+1} = (p_{t+1}, w_{t+1})` -- next period's price and renewable output.

**Transition function.**
- `R_{t+1} = R_t + x_t^{buy} - x_t^{sell}` (with efficiency losses in more detailed models)
- `p_{t+1}` follows a stochastic process (mean-reverting, with possible jumps)

**Contribution function.** `C(S_t, x_t) = p_t \cdot x_t^{sell} - p_t \cdot x_t^{buy}` (revenue from selling minus cost of buying).

**The Five Problem Variants and Dominant Policies.**

| Variant | Key Feature | Best Policy Class | Why |
|---|---|---|---|
| 1. Simple price thresholds | Prices have clear high/low regimes | **PFA** | A threshold rule (buy if `p_t < p^{low}`, sell if `p_t > p^{high}`) captures the essential structure |
| 2. Deterministic price cycles | Prices follow a known diurnal pattern with noise | **CFA** | Solve a deterministic optimization with a buffer parameter for noise; tune buffer via simulation |
| 3. Value of storage depends on level | Non-linear value of stored energy | **VFA** | The marginal value of storage changes with level; VFA captures `\bar{V}(R_t)` as a function of storage level |
| 4. Multi-day price forecasts available | Good forecasts but uncertain beyond horizon | **DLA** | Solve a multi-period deterministic optimization using forecasts; re-solve each period |
| 5. Complex interaction of features | All of the above | **Hybrid** | Combine elements from multiple classes |

**Policy details for each class.**

- **PFA example.** Buy when `p_t < \theta^{buy}`, sell when `p_t > \theta^{sell}`, do nothing otherwise. Thresholds `\theta^{buy}` and `\theta^{sell}` are tuned by simulation. May also depend on storage level: sell more aggressively when storage is nearly full.

- **CFA example.** Formulate a deterministic LP over a planning horizon using *expected* prices. Add a safety buffer: reduce apparent storage capacity by `\theta` percent to hedge against price uncertainty. Solve the LP; implement only the first period's decision. Tune `\theta` by evaluating over many simulated price paths.

- **VFA example.** Approximate the value function `\bar{V}_t(R_t, p_t)` as a piecewise-linear function of storage level `R_t` (possibly conditioned on price state `p_t`). At each step, solve:
  ```
  x_t = argmax_{x} [ C(S_t, x) + \gamma \bar{V}_{t+1}(R_{t+1}, p_{t+1}) ]
  ```
  Update `\bar{V}` using stochastic gradient methods. The piecewise-linear structure preserves convexity, enabling efficient optimization.

- **DLA example.** At each period, create a deterministic multi-period model using price forecasts `\hat{p}_{t+1}, ..., \hat{p}_{t+H}` for horizon `H`. Solve the resulting LP to get an optimal plan. Execute only the first period's decision `x_t`. Next period, update forecasts and re-solve. This is a form of *rolling-horizon* optimization, also called *model predictive control (MPC)* in the control literature.

**Key insight.** This is Powell's most important example because it demolishes the idea that any single method (RL, stochastic programming, MPC, rule-based control) is universally best. The optimal policy class depends on the problem structure: information quality, state dimensionality, value function shape, and forecast availability. The energy storage problem is simple enough to understand but rich enough to demonstrate all four classes.

**Citation.** Powell, W.B. and Meisel, S. (2016). "Tutorial on Stochastic Optimization in Energy -- Part I: Modeling and Policies" and "Part II: An Energy Storage Illustration." *IEEE Transactions on Power Systems*, 31(2), 1459-1467 and 1468-1475.

---

### 3.2 Broader Energy Applications

Beyond the canonical storage example, SDA applies across the energy sector:

**Unit commitment.** Deciding which power generation units to turn on/off over a planning horizon. This is a classic CFA application in the electric power industry: solve a deterministic mixed-integer program with reserve margins (tunable parameters) that hedge against demand and renewable uncertainty.

**Water reservoir management.** Sequential decisions about water release from a reservoir, balancing hydroelectric generation, flood control, irrigation, and environmental flows. State is water level; uncertainty is inflow (rainfall, snowmelt). VFA approaches have a long history here (value of water in the reservoir), and DLA (forecast-based release planning) is common in practice.

**Renewable integration.** Managing a portfolio of wind, solar, and conventional generation. Uncertainty in renewable output creates a need for storage, demand response, and backup generation. CFA approaches dominate practice (deterministic dispatch with reserve margins).

**Demand response.** Incentivizing consumers to shift electricity usage. The utility faces uncertainty about consumer response. PFA approaches (threshold-based incentives) and CFA approaches (optimize incentive schedule with response uncertainty buffers) are natural.

**Microgrid management.** A local energy system (campus, military base, island) with generation, storage, and load. All four policy classes appear depending on the microgrid's complexity and information environment.

---

## 4. Supply Chain Applications

### 4.1 Two-Agent Newsvendor Problem (SDAM Ch. 10)

**Narrative.** The classical newsvendor problem extended to two agents: a supplier (manufacturer) and a retailer. The retailer faces uncertain customer demand. The supplier faces uncertainty about the retailer's order (which depends on the retailer's beliefs about customer demand). Both make sequential decisions: the supplier sets production quantities and wholesale prices; the retailer sets order quantities and retail prices.

**State variables.**
- `I_t^{ret}` = retailer's inventory
- `I_t^{sup}` = supplier's inventory or production capacity
- `B_t^{ret}` = retailer's beliefs about customer demand distribution
- `B_t^{sup}` = supplier's beliefs about retailer ordering behavior
- Price history and contract terms

**Decision.**
- Retailer: order quantity `q_t^{ret}` (and possibly retail price)
- Supplier: production quantity `q_t^{sup}` and wholesale price `w_t`

**Exogenous information.** Customer demand `D_t` (revealed after retailer's stocking decision).

**Key insight.** Information asymmetry between agents creates strategic complexity. The supplier does not know the retailer's demand observations. This demonstrates how SDA handles multi-agent settings: each agent has their own state (including beliefs about the other agent) and their own policy. The framework remains the same -- it is the state definition that changes.

**Policies.**
- **PFA:** Newsvendor critical ratio solution as a parametric rule
- **CFA:** Parameterized ordering policy with safety stock levels tuned via simulation
- Contract design: buy-back contracts, revenue-sharing as CFA parameters

---

### 4.2 The Beer Game (SDAM Ch. 11)

**Narrative.** The Beer Game is a classic supply chain simulation, widely used in business schools since the 1960s (originated at MIT). It models a four-echelon supply chain:

```
Factory --> Distributor --> Wholesaler --> Retailer --> Customer
```

Each echelon must decide how much to order from the upstream echelon, facing uncertain downstream demand, non-trivial lead times (orders take multiple periods to arrive), and limited information (each echelon sees only its own inventory and incoming orders, not the full system state).

**State variables (per echelon `k`).**
- `I_t^k` = inventory on hand at echelon `k` (can be negative, representing backlog)
- `O_t^k` = outstanding orders placed but not yet received (pipeline inventory)
- `D_t^k` = most recent demand/orders received from downstream echelon
- Demand history (for forecasting)

**System-level state.** The full state is the concatenation across all four echelons, plus all in-transit orders. This is high-dimensional and only partially observable by each agent.

**Decision (per echelon).** `x_t^k` = order quantity to place with upstream echelon.

**Exogenous information.** `W_{t+1}` = customer demand `D_{t+1}^{ret}` at the retail level. This propagates upstream as orders, but with delays and distortion.

**Transition.** For each echelon `k`:
```
I_{t+1}^k = I_t^k + (shipments received from upstream) - (shipments sent downstream)
```
With lead time `L`: orders placed at time `t` arrive at time `t + L`.

**Contribution.** Minimize total cost = holding cost `h * max(I_t^k, 0)` + backlog cost `b * max(-I_t^k, 0)` across all echelons and time periods. Typically `b >> h` (backlogs are much more expensive than holding).

**Policies illustrated.**
- **PFA (base-stock policy):** Each echelon maintains a target inventory level `S^k`. Order quantity = `S^k - I_t^k - O_t^k` (order up to the base-stock level, accounting for pipeline). Simple, robust, widely used in practice.
- **CFA (parameterized ordering):** Order quantity = `\alpha * D_t^k + \beta * (S^k - I_t^k)` for tunable parameters `\alpha` (demand chasing) and `\beta` (inventory correction). This generalizes base-stock by adding a demand-chasing component.
- **VFA:** Approximate the value of being in each inventory state. Difficult due to high dimensionality (four echelons with pipeline stocks).
- **DLA:** Each echelon builds a forecast of future demand and solves a multi-period ordering plan. May or may not account for upstream capacity constraints.

**Key insight: the Bullwhip Effect.** The Beer Game's central lesson is that small fluctuations in customer demand get amplified as they propagate upstream. A 10% increase in retail demand can become a 50% spike at the factory level. This happens because each echelon independently over-reacts to demand signals, and lead times create information delays.

Poor policies (e.g., naive PFAs that over-react to demand changes) amplify the bullwhip effect. Good policies (e.g., well-tuned CFAs that smooth orders, or DLAs that share demand information across echelons) dampen it. The Beer Game is a classic teaching tool precisely because it demonstrates how policy design impacts system-level behavior in non-obvious ways.

**Extensions.**
- Information sharing: What if all echelons see actual customer demand? (Dramatically reduces bullwhip.)
- Stochastic lead times: Orders arrive with random delays.
- Capacity constraints: Factory has limited production capacity.
- Multiple products competing for capacity.
- AI/ML demand forecasting feeding into CFA or DLA policies.

---

### 4.3 General Inventory Management

Beyond the specific case studies, SDA provides a unified language for all inventory problems:

**(s, S) policies as PFAs.** The classic (s, S) policy -- order up to `S` when inventory drops below `s` -- is a PFA with two tunable parameters. The state is inventory level; the policy is a direct mapping from state to action.

**Safety stock as CFA parameter.** Many industrial inventory systems solve a deterministic replenishment optimization (e.g., an MRP/ERP system) and add safety stock as a buffer. The safety stock level is a CFA parameter tuned (explicitly or implicitly) against demand uncertainty.

**Multi-product, multi-location.** When many products share warehouse capacity or transportation, the problem becomes a resource allocation problem. CFA (parameterized joint ordering with shared constraints) and DLA (rolling-horizon joint optimization) are natural.

**Perishable inventory.** Products with expiration dates (food, blood, pharmaceuticals) add a state dimension (age of inventory) and a disposal decision. See Blood Management (Section 5.1).

---

## 5. Health Applications

### 5.1 Blood Management (SDAM Ch. 13)

**Narrative.** A hospital blood bank must manage inventory of blood products. Blood is:
- **Perishable:** Red blood cells expire after 42 days; platelets after 5 days.
- **Multi-typed:** Eight blood types (A+, A-, B+, B-, AB+, AB-, O+, O-) with specific compatibility rules.
- **Demand-uncertain:** Patient arrivals and blood type needs are stochastic.
- **Supply-uncertain:** Donations fluctuate (especially during holidays, disasters).

The blood bank must decide: how much to order from the regional blood center, and which units to issue first (oldest first to minimize waste? or freshest first to maximize remaining shelf life for the patient?).

**State variables.**
- `R_t = (r_t^{b,a})` for each blood type `b` and age category `a` -- inventory by type and age
- `D_t` = current demand (patients needing blood, by type and urgency)
- Possibly: day of week (affects demand patterns), upcoming scheduled surgeries

**Decision.**
- `x_t^{order}` = order quantities by blood type from regional center
- `x_t^{issue} = (x_t^{b,a \to p})` = assignment of blood units (type `b`, age `a`) to patients `p` (respecting compatibility)
- `x_t^{discard}` = units discarded due to expiration

**Exogenous information.** `W_{t+1}` = new patient arrivals (number and blood types needed), new donations received, emergency requests.

**Transition.**
- Inventory ages: units of age `a` become age `a+1`
- Units at maximum age are discarded
- Inventory adjusted for orders received and units issued

**Contribution.** Minimize: shortage costs (unmet demand, possibly weighted by urgency) + holding costs + wastage costs (expired units) + ordering costs.

**Policies illustrated.**
- **CFA (parameterized ordering with safety stocks):** Solve a deterministic replenishment model. Add safety stock parameters by blood type, tuned via simulation to balance shortage risk against wastage risk. This is how most real blood banks operate.
- **DLA (forecast-based planning):** Use surgical schedules and historical demand patterns to forecast blood needs over the next 1-2 weeks. Solve a multi-period allocation model. Re-solve daily.
- **PFA (issuance rules):** FIFO (oldest first) to minimize wastage, or type-specific matching rules to preserve rare types (e.g., never use O- for a patient who can receive other types).

**Key insight.** Multi-dimensional resource allocation with perishability creates a tension between ordering enough to avoid shortages and not ordering so much that units expire. The state space is large (inventory by type and age), but the problem has natural structure (compatibility rules, aging) that policies can exploit.

**Extensions.**
- Multi-hospital coordination (shared regional inventory)
- Emergency surge planning (mass casualty events)
- Integration with surgical scheduling (predictable demand component)
- Platelet management (much shorter shelf life, different dynamics)

---

### 5.2 Optimizing Clinical Trials (SDAM Ch. 14)

**Narrative.** A pharmaceutical company runs a clinical trial comparing `M` treatments. Patients arrive sequentially. For each patient, the trial designer must decide: which treatment arm to assign this patient to? And periodically: should we stop the trial early (because one treatment is clearly best, or clearly none work)?

This is structurally similar to the diabetes medication problem (Section 2.3) but with critical additional constraints:
- **Ethical obligation:** You cannot indefinitely assign patients to treatments you believe are inferior just to learn more.
- **Regulatory requirements:** FDA requires specific statistical evidence (p-values, confidence intervals).
- **Fixed budget:** Total number of patients is limited and expensive.
- **Multiple endpoints:** Treatments may have both efficacy and safety outcomes.

**State variables.**
- `B_t = (\mu_t^m, \sigma_t^m)_{m=1}^{M}` -- Bayesian posterior beliefs about each treatment's efficacy
- `n_t^m` = number of patients assigned to treatment `m` so far
- `N_t` = total patients enrolled so far (budget remaining = `N^{max} - N_t`)
- Possibly: interim safety signals, regulatory constraints

**Decision.**
- `x_t^{assign} \in \{1, ..., M\}` -- treatment assignment for the next patient
- `x_t^{stop} \in \{0, 1\}` -- whether to stop the trial early

**Exogenous information.** `W_{t+1}` = the treatment outcome for the assigned patient (efficacy score, adverse events).

**Policies illustrated.**
- **Knowledge gradient (PFA with beliefs):** Assign the next patient to the treatment that maximizes the expected improvement in the value of the best treatment's posterior mean. This one-step-lookahead on information value naturally balances exploration and exploitation.
- **Thompson sampling (PFA with beliefs):** Sample from each treatment's posterior; assign the patient to the treatment with the highest sample. This tilts allocation toward promising treatments while maintaining some exploration.
- **Response-adaptive randomization:** Allocate patients proportional to the posterior probability that each treatment is best. Widely used in modern clinical trials.

**Key insight.** This is *pure learning with ethical constraints*. Unlike the diabetes problem where the doctor treats one patient over time, here each patient is different and the ethical obligation is to each individual patient, not just the population. The explore-exploit tradeoff takes on moral weight: over-exploring wastes patients on inferior treatments; over-exploiting may miss a better treatment.

**Extensions.**
- **Biomarker-adaptive trials:** Patient features (biomarkers) affect treatment response -- contextual bandit structure.
- **Platform trials:** Treatments can be added or dropped mid-trial.
- **Multi-endpoint optimization:** Balance efficacy and safety.
- **Group sequential designs:** Pre-planned interim analyses with stopping rules (a CFA approach to trial design).

---

## 6. E-Commerce and Digital Applications

### 6.1 Ad-Click Optimization (SDAM Ch. 12)

**Narrative.** An online platform must decide which advertisement to show each user to maximize total clicks (or conversions, or revenue). There are `M` candidate ads. Each user arrives with context features (demographics, browsing history, device type). The platform does not know the true click-through rate (CTR) for each ad-context combination; it must learn through sequential experimentation.

**State variables.**
- `B_t = (\mu_t^{m,c}, \sigma_t^{m,c})` -- beliefs about CTR for each ad `m` in each context `c`
- User context `c_t` for the current user
- Possibly: budget constraints (remaining ad inventory), frequency caps

**Decision.** `x_t \in \{1, ..., M\}` -- which ad to display to the current user.

**Exogenous information.** `W_{t+1} \in \{0, 1\}` -- whether the user clicked on the displayed ad (Bernoulli outcome).

**Transition.** Bayesian update of beliefs for the (ad, context) pair that was shown:
- Use Beta-Bernoulli conjugate updating (or Gaussian approximation for computational efficiency)
- Update posterior: if click, shift mean upward; if no click, shift mean downward

**Policies illustrated.**
- **Thompson sampling (PFA):** For each candidate ad, sample a CTR from its posterior. Show the ad with the highest sampled CTR. Fast, effective, widely deployed at scale.
- **Upper confidence bounding / UCB (PFA):** Show the ad maximizing `\mu_t^{m,c} + \kappa \sigma_t^{m,c}`. The exploration bonus `\kappa \sigma` ensures under-explored ads get shown.
- **Knowledge gradient (PFA):** Show the ad that maximizes expected improvement in the best CTR estimate.
- **Epsilon-greedy (PFA):** With probability `1 - \epsilon`, show the best current ad; with probability `\epsilon`, show a random ad. Simple but wasteful exploration.

**Key insight.** This is online learning at industrial scale. The platform may serve millions of ad impressions per day, so:
1. Policies must be computationally cheap (Thompson sampling and UCB are O(1) per decision).
2. The explore-exploit tradeoff has direct financial consequences (each wasted impression costs revenue).
3. Context features create a combinatorial explosion of (ad, context) pairs -- motivating function approximation (neural networks predicting CTR as a function of ad and context features).

This is the *contextual bandit* problem, a generalization of the multi-armed bandit (Section 2.3) where the state includes contextual side information.

**Extensions.**
- **Bid optimization:** In auction-based ad platforms, the decision includes how much to bid.
- **Sequential engagement:** User sees multiple ads in a session; decisions are not independent.
- **Non-stationarity:** CTRs change over time (ad fatigue, seasonal trends).
- **Multi-objective:** Balance clicks, conversions, revenue, and user experience.
- **Cold-start:** New ads with no historical data; use content features and transfer learning.

---

## 7. Transportation and Logistics

Transportation is Powell's original research domain. He has worked on transportation optimization for 39+ years, and it was the application that motivated the universal SDA framework.

### 7.1 Dynamic Fleet Management

**Narrative.** A trucking company manages a fleet of vehicles across a network of locations. Loads (shipments) arrive randomly over time at various origins, each needing delivery to a destination by a deadline. The dispatcher must assign vehicles to loads, reposition empty vehicles, and manage driver schedules.

**State variables.**
- `R_t^v$ = location and status of each vehicle `v` (idle at city, en route, time until available)
- `L_t` = set of known loads (origin, destination, deadline, revenue)
- Driver hours remaining (hours-of-service regulations)
- Possibly: forecasts of future load arrivals

**Decision.** For each available vehicle: assign to a load, reposition empty to another city, or hold in place. This is a matching/assignment problem at each time step.

**Exogenous information.** `W_{t+1}` = new load arrivals, cancellations, traffic delays, vehicle breakdowns.

**Policies illustrated.**
- **PFA:** Nearest-available-vehicle dispatch rules; priority-based matching.
- **CFA:** Solve an assignment problem (match vehicles to loads) with modified costs that include a repositioning bonus for sending vehicles to high-demand areas. The bonus parameters are tuned via simulation.
- **VFA:** Estimate the future value of having a vehicle at each location and time. The value function `\bar{V}(v, location, time)` captures the expected future revenue of a vehicle positioned at a given location. Add this to the assignment optimization to account for downstream value.
- **DLA:** Forecast future load arrivals and solve a multi-period assignment model. Include anticipated future loads in the current matching decision.

**Key insight.** This is a resource allocation problem where the "resource" (vehicles) has a spatial and temporal dimension. Powell's work showed that VFA approaches (using value functions estimated via simulation) dramatically outperform myopic assignment, because they account for the repositioning value of vehicles. This insight led to the founding of Optimal Dynamics, Powell's company applying SDA to the trucking industry.

### 7.2 Dynamic Vehicle Routing

**Narrative.** A delivery service must route vehicles to serve customer requests that arrive throughout the day. Unlike static routing (all requests known upfront), requests arrive dynamically. The dispatcher must decide: serve this request now? Insert it into an existing route? Hold it for later?

**Policies.**
- **DLA (dominant):** At each decision point, solve a (possibly approximate) vehicle routing problem including all known requests and forecasted future requests. Re-solve when new requests arrive. This is the real-time re-optimization approach used by companies like Amazon, UPS, and food delivery platforms.
- **CFA:** Solve a routing optimization with modified time windows or capacity buffers to reserve capacity for expected future requests.

### 7.3 Google Maps as DLA

Powell frequently uses Google Maps navigation as an intuitive example of DLA. When you request directions:

1. Google Maps collects current traffic data across the road network (state `S_t`)
2. It forecasts future traffic conditions over your expected travel time (approximate model of `W_{t+1}, ..., W_{t+H}`)
3. It solves a deterministic shortest-path problem over this forecast (lookahead optimization)
4. It provides routing instructions (decision `x_t`)
5. As you drive, it continuously updates traffic data and re-solves (rolling-horizon re-optimization)

This is a DLA operating in real time on a dynamic stochastic network. It does not use VFA (no value function is stored). It does not use PFA (no simple rule like "always take highways"). It does not use CFA (no parameterized modification of a static model). It solves an approximate model of the future at each moment.

### 7.4 Optimal Dynamics

Optimal Dynamics is the company Powell co-founded to apply SDA to the trucking industry. It uses hybrid VFA+DLA approaches to:
- Match drivers to loads in real time
- Optimize repositioning of empty trucks
- Account for driver hours-of-service constraints
- Balance immediate revenue with long-term fleet positioning

This is the industrial-scale validation of the SDA framework: a company built entirely on the principle that no single policy class is best, and that the right approach combines multiple classes tuned to the specific problem structure.

---

## 8. Finance Applications

### 8.1 Portfolio Optimization Over Time

**Narrative.** An investor manages a portfolio of assets over multiple periods. Each period: observe asset returns, rebalance the portfolio (buy/sell assets), incur transaction costs.

**State.** Portfolio holdings `R_t = (r_t^1, ..., r_t^n)` (shares of each asset), beliefs about return distributions `B_t`, current wealth.

**Decision.** Trades `x_t = (x_t^1, ..., x_t^n)` subject to budget constraints and possibly short-selling restrictions.

**Policies.**
- **CFA:** Markowitz mean-variance optimization with modified inputs (shrinkage estimators, risk budgets as tunable parameters). This is the dominant approach in practice.
- **DLA:** Multi-period optimization with scenario trees or Monte Carlo.
- **PFA:** Constant-mix (rebalance to fixed target weights), momentum rules, mean-reversion rules.

### 8.2 Optimal Execution

**Narrative.** A trader must sell a large block of shares over multiple periods. Selling too fast moves the market against you (price impact); selling too slowly exposes you to price risk.

**State.** Shares remaining `R_t`, current price `p_t`, market conditions (volatility, spread).

**Decision.** How many shares to sell this period.

**Policies.**
- **PFA:** TWAP (time-weighted average price: sell equally over time), VWAP (volume-weighted: sell proportional to expected volume). Simple, widely used.
- **CFA:** Almgren-Chriss model with risk aversion parameter -- solve deterministic trajectory, adjust risk parameter.
- **VFA:** Approximate the value of remaining shares as a function of shares left and current price.

### 8.3 Options Pricing and Hedging

**Narrative.** Delta hedging an options portfolio over time under uncertain volatility and discrete rebalancing.

**Decision.** Hedge ratios at each rebalancing point.

**Policies.**
- **PFA:** Black-Scholes delta rule (a direct mapping from state to hedge ratio).
- **CFA:** Modified delta with transaction cost adjustment parameters.
- **VFA/DLA:** Deep hedging (neural network policies trained via simulation).

### 8.4 Asset-Liability Management

**Narrative.** A pension fund or insurance company must manage assets to meet future liabilities under uncertain returns and liability growth.

**Policies.**
- **CFA (dominant):** Solve a multi-period stochastic program with scenario trees; parameterize with risk constraints.
- **DLA:** Rolling-horizon optimization with updated forecasts.

---

## 9. Laboratory Sciences

### 9.1 Experimental Design

**Narrative.** A scientist must decide which experiments to run next, given a budget of `N` experiments. Each experiment tests a point in a (possibly high-dimensional) design space and returns a noisy measurement. The goal is to find the optimal design (e.g., the chemical formulation with highest yield).

**State.** Beliefs about the response surface `B_t` (e.g., a Gaussian process posterior).

**Decision.** Where in the design space to sample next.

**Policies.**
- **Knowledge gradient (PFA):** Sample where expected improvement in the best known point is maximized.
- **Expected improvement (PFA):** Sample where the expected improvement over the current best observation is largest. This is the standard Bayesian optimization acquisition function.
- **Upper confidence bound (PFA):** Sample where `\mu_t(x) + \kappa \sigma_t(x)` is maximized.

### 9.2 Drug Discovery

**Narrative.** Pharmaceutical companies must decide which candidate compounds to synthesize and test, given that synthesis is expensive and testing is time-consuming. The design space is combinatorially large (millions of possible molecules).

**Policies.** Bayesian optimization (PFA with knowledge gradient or expected improvement) over molecular descriptors. Increasingly combined with neural network surrogate models.

### 9.3 Materials Science Optimization

**Narrative.** Finding optimal material compositions (alloys, polymers, catalysts) through sequential experimentation. Similar structure to drug discovery but in a continuous or mixed design space.

**Policies.** Bayesian optimization with Gaussian process surrogates. Knowledge gradient and expected improvement are the workhorses.

### 9.4 Sequential Experiment Planning

**Narrative.** General framework for any scientific investigation where:
1. Experiments are expensive (time, money, or both)
2. Results are noisy
3. The goal is to maximize information gained or find an optimum within a budget

This is the *ranking and selection* problem in the simulation literature and the *Bayesian optimization* problem in machine learning. SDA provides the unifying framework: the state is the belief, the decision is which experiment to run, the uncertainty is the experimental outcome, and the policy is an acquisition function (PFA).

---

## 10. Cross-Domain Patterns

### 10.1 Application Type to Dominant Policy Class

| Application Type | Dominant Class | Rationale |
|---|---|---|
| Pure learning (bandits, experimental design) | **PFA** | State is beliefs; policies are direct functions of beliefs (Thompson sampling, KG, UCB) |
| Optimal stopping (asset selling) | **PFA / VFA** | Threshold rules (PFA) or exact value iteration (VFA) |
| Single-resource storage (energy battery) | **All four** | Depends on information structure, price dynamics, forecast quality |
| Inventory management (newsvendor, beer game) | **PFA / CFA** | (s,S) rules (PFA); safety stock buffers (CFA) |
| Multi-resource allocation (fleet management) | **VFA / CFA** | VFA captures spatial value; CFA modifies assignment cost |
| Dynamic routing / navigation | **DLA** | Real-time re-optimization over forecasts |
| Clinical trials / health | **PFA** | Knowledge gradient, Thompson sampling with belief states |
| Financial planning | **CFA** | Parameterized optimization models (Markowitz, ALM) |
| Multi-echelon supply chain | **CFA / DLA** | Parameterized ordering (CFA); forecast-based planning (DLA) |

### 10.2 Common Structural Patterns

**Pattern 1: Pure learning problems use PFAs with belief states.**
When the only "resource" is information, the state is the belief state (posterior distributions), and good policies are direct functions of beliefs: Thompson sampling, knowledge gradient, UCB. There is no physical resource to optimize over, so CFA and DLA add little. VFA can help (value of information) but is often intractable.

**Pattern 2: Resource allocation problems use CFA or DLA.**
When there are physical resources (inventory, vehicles, energy, money) to allocate, CFA and DLA dominate because they leverage the structure of the allocation problem (LP, MIP, network flow). CFA modifies the allocation model to hedge against uncertainty. DLA extends it over a forecast horizon.

**Pattern 3: Hybrid problems combine classes.**
Many real problems have both learning and allocation components. Examples: fleet management with demand learning (VFA for spatial value + CFA for assignment), clinical trials with capacity constraints (PFA for treatment selection + CFA for enrollment planning). The SDA framework naturally accommodates hybrids by combining policy classes.

**Pattern 4: Information quality determines the best DLA.**
When good forecasts are available (weather, scheduled surgeries, traffic patterns), DLA is powerful. When forecasts are poor, DLA degrades and PFA or VFA may be better. The quality of the approximate model in the lookahead is critical.

**Pattern 5: State dimensionality determines VFA feasibility.**
VFA requires approximating a function over the state space. For low-dimensional states (energy storage level, asset price), VFA is tractable and effective. For high-dimensional states (multi-echelon inventory, large fleet), VFA requires aggressive approximation (basis functions, neural networks) and may be less reliable than CFA or DLA.

### 10.3 How to Adapt a Case Study to a New Domain

When a user presents a new sequential decision problem, use the following procedure:

1. **Identify the narrative.** What decisions are being made? What is uncertain? What is the objective?

2. **Map to the five elements.**
   - State: What do you know when making a decision? Decompose into physical state `R_t`, information state `I_t`, and belief state `B_t`.
   - Decision: What are the feasible actions? What constraints apply?
   - Exogenous information: What new information arrives after each decision?
   - Transition: How does the state evolve?
   - Objective: Maximize/minimize what, over what horizon?

3. **Find the closest case study.** Use the table in Section 10.1 to identify the application type. Find the matching case study in Sections 2-9. The policy classes that work for the case study are strong candidates for the new problem.

4. **Assess information structure.** Is learning central (beliefs dominate state)? Is resource allocation central (physical state dominates)? Are good forecasts available? This determines which policy classes to prioritize.

5. **Design candidate policies from each relevant class.** For each candidate class:
   - PFA: What would a simple rule look like? What state features would it use?
   - CFA: What deterministic optimization model applies? What parameters would you add to handle uncertainty?
   - VFA: Can you approximate the value function tractably? What are the state dimensions?
   - DLA: What lookahead model would you solve? What forecast inputs would it use?

6. **Evaluate by simulation.** Generate or obtain stochastic scenarios. Evaluate each candidate policy. Tune parameters. Compare.

7. **Iterate.** Refine the best-performing policies. Consider hybrids. Test robustness.

---

## 11. Citations and References

### Primary Textbooks

- Powell, W.B. (2022). *Sequential Decision Analytics and Modeling: Modeling with Python*. Now Publishers. (Referred to as **SDAM** throughout this document.)
  - Chapter 2: Asset Selling Problem
  - Chapter 3: Adaptive Market Planning
  - Chapter 4: Learning the Best Diabetes Medication (Multi-armed bandits)
  - Chapter 5: Stochastic Shortest Path -- Static
  - Chapter 6: Stochastic Shortest Path -- Dynamic
  - Chapters 8-9: Energy Storage I and II
  - Chapter 10: Two-Agent Newsvendor Problem
  - Chapter 11: The Beer Game (Multi-echelon Supply Chain)
  - Chapter 12: Ad-Click Optimization
  - Chapter 13: Blood Management
  - Chapter 14: Clinical Trial Optimization

- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions*. Wiley. (Referred to as **RLSO**.)
  - Provides the theoretical foundations and additional application domains.

### Key Journal Articles

- Powell, W.B. and Meisel, S. (2016). "Tutorial on Stochastic Optimization in Energy -- Part I: Modeling and Policies." *IEEE Transactions on Power Systems*, 31(2), 1459-1467.
- Powell, W.B. and Meisel, S. (2016). "Tutorial on Stochastic Optimization in Energy -- Part II: An Energy Storage Illustration." *IEEE Transactions on Power Systems*, 31(2), 1468-1475.
- Powell, W.B. (2019). "A Unified Framework for Stochastic Optimization." *European Journal of Operational Research*, 275(3), 795-821.
- Powell, W.B. (2020). "From Reinforcement Learning to Optimal Control: A Unified Framework for Sequential Decisions." In *Handbook of Reinforcement Learning and Control*, Springer.

### Application-Specific References

- **Transportation and fleet management:** Powell, W.B. (1996 onward). Extensive body of work on dynamic fleet management. Powell co-founded Princeton Transportation Consulting Group and later Optimal Dynamics.
- **Multi-armed bandits and knowledge gradient:** Frazier, P., Powell, W.B., and Dayanik, S. (2008). "A Knowledge-Gradient Policy for Sequential Information Collection." *SIAM Journal on Control and Optimization*, 47(5), 2410-2439.
- **Bayesian optimization and ranking and selection:** Powell, W.B. and Ryzhov, I.O. (2012). *Optimal Learning*. Wiley. Covers the knowledge gradient, expected improvement, and other learning policies in depth.
- **Beer Game and supply chain:** Sterman, J.D. (1989). "Modeling Managerial Behavior: Misperceptions of Feedback in a Dynamic Decision Making Experiment." *Management Science*, 35(3), 321-339. (Original bullwhip effect analysis.)
- **Clinical trials:** Berry, D.A. (2006). "Bayesian Clinical Trials." *Nature Reviews Drug Discovery*, 5, 27-36.

### Related Frameworks

- **Markov Decision Processes (MDPs):** Puterman, M.L. (2014). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
- **Stochastic Programming:** Birge, J.R. and Louveaux, F. (2011). *Introduction to Stochastic Programming*. Springer.
- **Model Predictive Control:** Rawlings, J.B., Mayne, D.Q., and Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design*. Nob Hill Publishing.
- **Reinforcement Learning:** Sutton, R.S. and Barto, A.G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.

---

## Usage Notes for This Skill

When applying this skill:

1. **Start with the narrative.** Users describe problems in natural language. Match their description to the closest application narrative above, then use the corresponding model structure as a template.

2. **Use the five-element model as a checklist.** For any new problem, systematically identify: state, decision, exogenous information, transition, objective. If any element is unclear, ask the user to clarify.

3. **Let the problem structure choose the policy class.** Do not default to any single method. Use Section 10.1 to identify candidate policy classes based on the application type, then evaluate multiple candidates.

4. **Energy storage is the master example.** If you need to illustrate why all four policy classes matter, use the energy storage variants from Section 3.1.

5. **The Beer Game is the best teaching example for supply chains.** It is widely known, highly intuitive, and demonstrates bullwhip effects and policy impact clearly.

6. **For learning problems, default to knowledge gradient or Thompson sampling.** These are Powell's preferred PFA approaches for multi-armed bandits and Bayesian optimization problems.

7. **For resource allocation problems, start with CFA.** Most industrial optimization already uses deterministic models. CFA extends these naturally by adding tunable parameters -- the lowest-friction path to incorporating uncertainty.
