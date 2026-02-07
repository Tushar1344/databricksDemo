# SDA Reasoning and Discovery Skill

## 1. Purpose

This skill enables using the **Sequential Decision Analytics (SDA)** framework to REASON about new problems, DISCOVER new application domains, and THINK through sequential decision problems creatively. It is a meta-cognitive skill for applying Powell's universal framework to novel situations.

**Core objectives:**

- Enable creative application of SDA thinking to NEW problem domains beyond textbook examples
- Provide a structured reasoning framework for identifying sequential decision problems "in the wild"
- Help generate novel research questions and practical applications by mapping real-world situations to the five elements (state, decision, exogenous information, transition, objective)
- Move beyond standard examples (inventory, routing, bandits) to real-world problem discovery across every discipline

When this skill is invoked, the agent should adopt the mindset of a researcher trained in Powell's SDA framework, looking at problems through the lens of sequential decisions under uncertainty and systematically mapping them to the universal modeling structure.

---

## 2. The SDA Lens: Seeing Sequential Decisions Everywhere

Almost any human process involves sequential decisions under uncertainty. The key insight is learning to recognize them. Once you internalize the SDA lens, you begin to see sequential decision problems in every domain of human activity.

### The Recognition Pattern

Ask three questions about any process or situation:

1. **Is there a sequence of decisions being made over time?** (not a single one-shot choice)
2. **Is there uncertainty being revealed between decisions?** (new information arrives)
3. **Would better decisions improve outcomes?** (there is room for optimization)

If the answer is **yes to all three**, this is an SDA problem.

### Examples of Non-Obvious SDA Problems

Many problems that people do not traditionally think of as "optimization" or "sequential decision" problems are, in fact, rich SDA problems:

- **Career planning:** Sequential job and education choices with uncertain outcomes, evolving skills, and changing market conditions. Each career move updates beliefs about personal preferences and aptitudes.
- **Dating and matching:** Sequential choices about whom to date, with uncertain compatibility revealed through interaction. Classic secretary-problem structure with richer state.
- **Software development:** Feature prioritization with uncertain user value. Each release reveals information about what users actually want. Sprint planning is a sequential resource allocation problem.
- **Scientific research:** Which experiments to run next, given current knowledge and budget. A pure learning problem with expensive observations and belief-dependent decisions.
- **Content creation:** What to write, film, or produce next given audience feedback. Creators update beliefs about audience preferences and optimize engagement over time.
- **Urban planning:** Infrastructure investments over decades with uncertain population growth, technology change, and climate impacts. Extremely long horizon with irreversible decisions.
- **Climate policy:** Emission reduction targets with uncertain climate response, uncertain technology costs, and geopolitical dynamics. Multi-agent with cooperative and competitive elements.
- **Cybersecurity:** Defense resource allocation with uncertain attack patterns. An adversarial sequential game where the attacker's strategy is partially observable.
- **Farming:** Crop selection, planting timing, irrigation, and harvest decisions with weather uncertainty, pest uncertainty, and price uncertainty. Multi-scale temporal decisions.
- **Personal health:** Diet, exercise, medication, and screening choices with uncertain health outcomes. Belief states include estimated disease risk that updates with each test.

---

## 3. The Discovery Framework

Use this six-step framework to systematically analyze any candidate SDA problem.

### Step 1: Identify the Decision Sequence

Answer these questions to establish the temporal structure:

- **What decisions are made?** List the concrete choices at each point in time.
- **How often are decisions made?** What is the decision epoch frequency?
- **Who makes them?** A person, an algorithm, an organization, a committee?
- **What triggers a new decision?** Is it clock-driven (every hour/day/month), event-driven (when inventory hits zero, when a patient arrives), or continuous?
- **What is the time scale?** Milliseconds for high-frequency trading, seconds for autonomous driving, hours for dispatching, days for inventory, years for infrastructure, decades for climate. The time scale profoundly affects which policy classes are viable.

### Step 2: Map the Uncertainties

Uncertainty is the core of what makes sequential decisions hard. Characterize it precisely:

- **What is unknown when decisions are made?** List every source of uncertainty.
- **How does uncertainty resolve over time?** All at once? Gradually? In response to decisions?
- **What type of uncertainty is it?**
  - *Environment parameters:* Unknown but fixed quantities (e.g., a drug's true efficacy)
  - *Future events:* Stochastic processes that unfold (e.g., demand, weather)
  - *Consequences of actions:* Outcome of a choice is uncertain (e.g., will this treatment work for this patient?)
- **Can the agent's decisions INFLUENCE what it learns?** This is the critical distinction between active and passive learning. If yes, there is an exploration-exploitation tradeoff.
- **Is uncertainty exogenous (independent of decisions) or endogenous (influenced by decisions)?**

### Step 3: Define the State Space

The state variable is the information needed to model the system from time t onward. It typically has three components:

- **Physical/resource state (R_t):** Tangible quantities like inventory levels, cash on hand, fleet positions, resource stocks, infrastructure in place.
- **Belief state (B_t):** What we currently believe about unknowns. This could be a posterior distribution over parameters, a set of estimated probabilities, or a confidence interval. The belief state is what makes learning problems tractable within the SDA framework.
- **Contextual information (I_t):** Features, covariates, and observable environmental conditions that affect the current decision but are not directly controlled. Examples: current weather, market conditions, patient demographics.

The full state is S_t = (R_t, B_t, I_t). Identify which components are present and their dimensionality.

### Step 4: Characterize the Problem Type

Use this classification grid to place the problem in the SDA taxonomy. For each dimension, select the option that best fits:

| Dimension | Options | Notes |
|-----------|---------|-------|
| **Learning vs. Resource allocation** | Pure learning / Pure resource allocation / Hybrid | Pure learning: decisions exist only to gather info. Pure resource: no learning needed. Hybrid: most real problems. |
| **Horizon** | Finite / Infinite / Event-driven | Finite: known end time. Infinite: ongoing. Event-driven: ends when condition met. |
| **Decision type** | Scalar / Vector / Combinatorial | Scalar: one number. Vector: multiple continuous. Combinatorial: discrete choices from large set. |
| **State dimension** | Low (1-5) / Medium (5-50) / High (50+) | Critical for determining VFA viability. |
| **Uncertainty type** | Parametric / Distributional / Model | Parametric: known distribution family, unknown params. Distributional: unknown distribution. Model: unknown structure. |
| **Information structure** | Full / Partial / Adversarial | Full: observe entire state. Partial: noisy or incomplete. Adversarial: opponent hides information. |
| **Multi-agent?** | Single / Cooperative / Competitive | Single: one decision-maker. Cooperative: aligned objectives. Competitive: conflicting objectives. |

### Step 5: Suggest Policy Approaches

Based on the characterization from Step 4, reason about which policy class(es) are most promising. Use these heuristics:

- **Policy Function Approximation (PFA):** Best when the problem is simple and low-dimensional, when good domain intuition exists for what a good policy looks like, or when you can parameterize a rule (e.g., order-up-to level, threshold rule, priority ranking). PFA is also the natural home for lookup-table policies and parametric rules tuned by simulation.
- **Cost Function Approximation (CFA):** Best when there is an existing deterministic optimization model (LP, MIP, assignment) that can be modified with tunable parameters to handle uncertainty. CFA embeds learning into an optimization model by adjusting costs, constraints, or parameters. Powerful for high-dimensional resource allocation.
- **Value Function Approximation (VFA):** Best when there is a clear state-value structure, the state space is moderate-dimensional, and Bellman's equation provides useful structure. Includes approximate dynamic programming and deep reinforcement learning. Requires careful approximation architecture design.
- **Direct Lookahead Approximation (DLA):** Best when a good forward model (simulator) exists, planning ahead is valuable, and the decision benefits from reasoning about future scenarios. Includes stochastic programming, model predictive control, Monte Carlo tree search, and rollout policies.
- **Hybrid approaches:** Many real problems benefit from combining classes. For example, CFA with VFA-tuned parameters, or DLA with a VFA terminal value approximation. Complex, multi-faceted problems almost always need hybrids.

### Step 6: Identify Novel Aspects

This is where discovery happens. Ask:

- **What makes this problem different from textbook examples?** Every real problem has wrinkles that standard formulations miss.
- **Are there new types of constraints?** Multi-objective tradeoffs? Fairness requirements? Safety constraints? Robustness needs? Interpretability demands?
- **Does it require new types of belief states?** Beliefs over graphs, beliefs over functions, beliefs over causal structures?
- **Could it inspire new policy designs?** Sometimes a novel application domain demands a new algorithmic idea.
- **What data is available?** Real-world data availability often shapes what is feasible in ways that theory ignores.

---

## 4. Reasoning Templates

Use these structured templates when analyzing problems through the SDA lens.

### Template A: "Is This an SDA Problem?"

Given a description of a real-world situation, systematically determine if and how it maps to the SDA framework.

**Procedure:**

1. **Identify the sequential nature.** What decisions repeat over time? What is the time scale? Is there a natural ordering? If the situation involves only a single one-shot decision with no future consequences, it is NOT an SDA problem (it is a static optimization or decision analysis problem).

2. **Identify the uncertainty.** What information is missing when decisions are made? How and when does it arrive? If all information is known in advance, it is NOT an SDA problem under uncertainty (it is a deterministic sequential problem, still potentially hard but different in character).

3. **Sketch the five elements.**
   - State S_t: What does the decision-maker know at time t?
   - Decision x_t: What choices are available?
   - Exogenous information W_{t+1}: What new information arrives?
   - Transition S_{t+1} = S^M(S_t, x_t, W_{t+1}): How does the state evolve?
   - Objective: What is being optimized? Over what horizon? Risk-neutral or risk-sensitive?

4. **Classify the problem type** using the grid from Step 4 of the Discovery Framework.

5. **Suggest initial approaches.** Based on the classification, which policy classes merit investigation first?

### Template B: "What Policy Class Fits?"

Given a well-defined SDA problem, reason through which policy class(es) are most promising.

**Procedure:**

1. **What domain knowledge exists?** If practitioners have strong intuitions about what good policies look like (e.g., "reorder when inventory drops below X"), this favors PFA or CFA. Encode this knowledge.

2. **Is there a natural optimization model?** If the problem without uncertainty maps to a known optimization formulation (LP, MIP, network flow, assignment), CFA is powerful. Modify the deterministic model with tunable parameters.

3. **Is the state space tractable for value functions?** Count the effective state dimensions. Below roughly 5-10 dimensions, exact or tabular VFA may work. Up to 50-100 dimensions, approximate VFA with basis functions or neural networks is feasible. Beyond that, VFA becomes very challenging and CFA or DLA may be better.

4. **Is planning ahead valuable?** If the problem has significant multi-period structure (decisions now strongly affect future options), DLA methods that explicitly plan ahead are valuable. If the problem is approximately myopic (current reward dominates), simpler PFA may suffice.

5. **What computational budget exists?** PFA is cheapest at decision time (evaluate a function). CFA requires solving an optimization at each step. VFA requires training but is fast at decision time. DLA requires solving optimization problems at decision time and may be expensive.

6. **What is the cost of exploration?** In safety-critical domains (healthcare, autonomous driving), pure exploration is dangerous. Favor methods with bounded risk: conservative PFA, robust CFA, constrained DLA. In low-stakes domains (content recommendation, A/B testing), exploration is cheap and VFA or Bayesian PFA can be aggressive.

### Template C: "How Would Powell Frame This?"

Adopt Powell's perspective to analyze a problem that may have been studied by other communities under different names and notations.

**Procedure:**

1. **What community would traditionally study this?** Identify the academic home: RL, control theory, stochastic programming, OR, economics, statistics, etc.

2. **What notation would they use?** RL uses (s, a, r, s'). Control uses (x, u, w). Stochastic programming uses scenarios and stages. Each community's notation reveals what they emphasize and what they hide.

3. **How would the SDA universal framework represent it?** Map to (S_t, x_t, W_{t+1}, S^M, objective). The universal framework forces you to be explicit about ALL components.

4. **What do the five elements look like?** Write them out concretely. Pay special attention to the state variable: does the traditional formulation include the belief state? Often it does not.

5. **Would the traditional community miss any policy class?** This is the key payoff. RL researchers rarely consider CFA. Stochastic programmers rarely consider PFA. Control theorists may miss VFA with learned value functions. By considering all four classes, you may find that the best approach is one the traditional community never tried.

### Template D: "Generate New Applications"

For a given domain (e.g., agriculture, education, space exploration), systematically generate new SDA applications.

**Procedure:**

1. **List all sequential decision processes in this domain.** Be exhaustive. Interview domain experts if possible. Think about decisions at every time scale and organizational level.

2. **For each process, identify the key uncertainties.** What makes the decision hard? What would a clairvoyant do differently?

3. **Prioritize by: impact x feasibility x novelty.**
   - Impact: How much value would better decisions create?
   - Feasibility: Is data available? Can we build a model? Can we test solutions?
   - Novelty: Has this been studied before? Is there a gap in the literature?

4. **For the top candidates, sketch the five elements.** Be concrete. Define the state variables, decision variables, exogenous information, transition function, and objective precisely enough to start modeling.

5. **Identify what makes each unique from existing literature.** What new methodological challenges arise? Could solving this problem advance the field, not just the application?

---

## 5. Cross-Domain Transfer Patterns

One of the most powerful aspects of the SDA framework is recognizing that problems in completely different domains can have identical mathematical structure. When you solve a problem in one domain, the solution transfers to all isomorphic problems.

**Key structural equivalences:**

- **Energy storage <-> Inventory management:** Both involve managing a buffer (battery/warehouse) with stochastic supply and demand, capacity constraints, and buying/selling decisions. The physics differ but the math is identical.
- **Clinical trials <-> A/B testing <-> Ad optimization:** All are sequential learning problems where you allocate samples across alternatives to identify the best one while maximizing cumulative reward. Same bandit structure, different stakes and time scales.
- **Fleet routing <-> Network design:** Both involve decisions on graph structures with uncertain demands and costs. Routing is operational (short horizon), design is strategic (long horizon), but the combinatorial structure is shared.
- **Portfolio management <-> Resource allocation:** Both involve distributing a budget across options with uncertain returns. Financial portfolios have continuous rebalancing; resource allocation may have discrete choices. But the allocation-under-uncertainty structure is the same.
- **Epidemic control <-> Invasive species management:** Both involve spatial spread of an undesirable agent, sequential intervention decisions, and uncertain dynamics. Control actions (vaccination/eradication) are localized and resource-constrained.
- **Compiler optimization <-> Chemical process control:** Both involve a sequence of transformations applied to a state, with uncertain outcomes and a quality metric. The "state" is code in one case and a chemical mixture in the other.

**Transfer principle:** When you encounter a solved SDA problem in one domain, always ask: *Where else does this mathematical structure appear?* The answer often reveals new applications where existing algorithms can be directly applied or adapted.

---

## 6. Novel Research Directions

The SDA framework opens up numerous frontier research questions. When reasoning about a new problem, consider whether it touches any of these open areas:

- **Multi-agent SDA:** How do multiple sequential decision-makers interact? Competitive settings (game theory meets SDA) and cooperative settings (team decision theory, mechanism design for sequential problems) both present open challenges.
- **Fairness in sequential decisions:** How do we ensure sequential decisions do not systematically disadvantage certain groups? Fairness constraints interact with learning in subtle ways: exploration itself can be unfair.
- **Causal reasoning within SDA:** Standard SDA assumes a transition function but does not reason about causality explicitly. Integrating causal inference (do-calculus, structural causal models) with SDA could enable better generalization and transfer.
- **Foundation models for SDA:** Can large language models or other foundation models serve as policy approximators? As world models for DLA? As prior knowledge encoders for belief states? This is a nascent but high-potential direction.
- **SDA for AI alignment:** The problem of aligning AI systems with human values is itself a sequential decision problem under uncertainty (uncertainty about human preferences, sequential updates as we learn more). Meta-level SDA.
- **Meta-learning across SDA problems:** Can we learn to solve new SDA problems faster by leveraging experience from previously solved ones? Learning the structure of problems, not just their solutions.
- **SDA in partially observable environments (POMDP integration):** When the physical state is not fully observed, the belief state must encompass beliefs about both parameters AND hidden state. This dramatically increases complexity.
- **SDA with human-in-the-loop:** Real systems often have human decision-makers who are boundedly rational, have preferences about the decision process itself, and whose trust in automated systems evolves over time. Modeling the human as part of the system is an SDA problem within an SDA problem.

---

## 7. The "15 Communities" Perspective

Different academic communities have developed different tools for sequential decision problems. Each community tends to favor certain policy classes and may be unaware of others. Understanding this landscape helps identify blind spots and opportunities.

| Community | Primary Focus | Typical Policy Class | Common Blind Spots |
|-----------|--------------|---------------------|-------------------|
| **Reinforcement learning** | Learning value functions from interaction | VFA (Q-learning, actor-critic, deep RL) | May miss CFA; often ignores problem structure that could be exploited |
| **Stochastic programming** | Optimization under scenarios | DLA (scenario trees, two-stage models) | May miss PFA; computationally limited by scenario explosion |
| **Optimal control** | Continuous-state control systems | DLA (MPC) and VFA (Bellman/HJB) | May miss CFA; often assumes known dynamics |
| **Simulation optimization** | Tuning parameters via simulation | PFA (parameter tuning of policy rules) | May miss VFA; limited to parameterized policy space |
| **Bandits and online learning** | Sequential allocation with learning | PFA (UCB, Thompson sampling, epsilon-greedy) | May miss CFA and DLA; often assumes simple action spaces |
| **Dynamic programming** | Exact recursive optimization | VFA (exact or approximate Bellman) | May miss CFA; limited by curse of dimensionality |
| **Bayesian optimization** | Optimizing expensive black-box functions | PFA (acquisition functions based on belief state) | May miss VFA; focused on continuous, low-dimensional spaces |
| **Robust optimization** | Worst-case guarantees | DLA (minimax formulations) | May miss PFA; can be overly conservative |
| **Markov decision processes** | Sequential decisions with known model | VFA (policy/value iteration) | May miss CFA; assumes model is known and state is observable |
| **Model predictive control** | Receding-horizon planning | DLA (deterministic or scenario-based lookahead) | May miss VFA for terminal values; often ignores learning |
| **Operations research** | Practical optimization in organizations | CFA (modified deterministic models) and PFA (heuristics) | May miss VFA; sometimes too focused on one-shot optimization |
| **Machine learning** | Learning from data, prediction | VFA (deep RL, imitation learning) | May miss CFA and domain-specific PFA; may over-parameterize |
| **Decision analysis** | Structuring decisions under uncertainty | DLA (decision trees, influence diagrams) | May miss PFA and VFA; limited to small problems by tree explosion |
| **Economics** | Structural models of agents and markets | VFA (structural estimation of dynamic models) | May miss CFA; computationally constrained by estimation requirements |
| **Management science** | Analytical models for business decisions | PFA (analytical rules, newsvendor-type policies) | May miss VFA and DLA; may over-simplify dynamics for tractability |

**How to use this table:** When analyzing a new problem, identify which community would traditionally "own" it, then deliberately consider policy classes that community typically ignores. This systematic cross-pollination is one of the greatest practical benefits of the SDA framework.

---

## 8. Practical Exercises for Discovery

Use these exercises to practice and sharpen SDA reasoning skills:

### Exercise 1: Daily Life as SDA
Pick a daily activity (commuting, cooking, shopping, exercising). Frame it as a full SDA problem. Define all five elements. What is your current policy? Is it a PFA, CFA, VFA, or DLA? Could you do better with a different policy class?

### Exercise 2: News Article Mapping
Read a news article about a business challenge, policy debate, or technological problem. Map it to the five elements of the SDA framework. Identify the key uncertainties. What policy class would you recommend? Write a one-paragraph "SDA analysis" of the article.

### Exercise 3: RL Benchmark Rethinking
Take a solved RL benchmark environment (CartPole, MountainCar, Atari games, MuJoCo tasks). These are typically solved with VFA (deep RL). Ask: Would CFA or PFA work better? Could you design a simple parametric policy (PFA) that matches deep RL performance? Could you formulate a deterministic optimization with tuned parameters (CFA)?

### Exercise 4: Organizational Problem Identification
Identify a sequential decision problem in your organization or institution. Classify it using the taxonomy from Step 4 of the Discovery Framework. Who currently makes this decision? What policy class are they implicitly using? What class might work better?

### Exercise 5: Structural Twins
Find two problems from completely different domains (e.g., one from healthcare and one from logistics) that have identical mathematical structure. Write out the five elements for both and show the mapping between them. What solution techniques transfer?

---

## 9. Citations

These references provide the theoretical foundation for the SDA framework and the reasoning patterns in this skill:

- Powell, W.B. (2022). *Sequential Decision Analytics and Modeling: Modeling with Python.* Now Publishers. Especially Ch. 7 (Applications Revisited) for the breadth of SDA applications and the classification of problems across domains.
- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions.* John Wiley & Sons. Especially Ch. 1 (The Challenges of Sequential Decision Problems) for the motivation of the universal framework and the four policy classes.
- Powell, W.B. (2019). "A unified framework for stochastic optimization." *European Journal of Operational Research*, 275(3), 795-821. The journal article presenting the unifying perspective across the 15 communities.
- CASTLE Lab, Princeton University. "From the Jungle of Stochastic Optimization to Sequential Decision Analytics." Available at: castle.princeton.edu/jungle/ — A comprehensive guide to navigating the fragmented landscape of sequential decision-making research.
