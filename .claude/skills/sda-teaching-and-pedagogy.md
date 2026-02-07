# SDA Teaching and Pedagogy

## Purpose
This skill provides guidance on how to **teach Sequential Decision Analytics (SDA)** effectively, based on Warren B. Powell's pedagogical approach developed over 39 years at Princeton University. It covers course design, teaching methods, common misconceptions, and audience adaptation.

---

## 1. Powell's Pedagogical Philosophy

### Core Principles
- **"Teach by example"** — Use concrete applications to illustrate abstract framework, not theory-first
- **"Model first, then solve"** — Standard in deterministic optimization, but novel for stochastic problems. Students learn to MODEL a problem using the five elements BEFORE choosing a solution method
- **Every chapter follows the same outline:** Narrative → Model (five elements) → Uncertainty → Policies → Evaluation → Extensions
- **All four policy classes must be taught** — The academic community over-emphasizes the most complex policies (VFA/deep RL) that are rarely used in practice. Students should learn ALL four so they can choose what works best
- **No prerequisites in stochastic optimization** — SDA can be taught to undergraduates with basic probability and optimization background
- **Customizable by domain** — The same framework applies to any field; swap examples to match your audience

### Key Quote
> "The academic community insists on researching and teaching the most complex policies rarely used in practice." — Warren B. Powell

### Why This Matters
SDA should be taught **alongside courses in optimization and machine learning**, not as an advanced specialization. It is a fundamental skill for anyone making decisions under uncertainty — which is everyone.

---

## 2. Course Structure: The 11-Topic Outline

Powell designed an 11-topic course outline, published in a monograph on teaching optimization. Each topic maps to SDAM chapters:

### Topic 1: Introduction to Sequential Decision Problems (Week 1)
- **Content:** The decision → information → decision → information pattern
- **Motivating examples:** Everyday decisions (routing, shopping, career), then professional (energy, health, logistics)
- **Goal:** Students recognize sequential decisions everywhere
- **Activity:** Students identify 3 sequential decision problems from their daily life

### Topic 2: The Universal Modeling Framework (Week 2)
- **Content:** Five core elements (S_t, x_t, W_t, S^M, objective)
- **SDAM Reference:** Chapter 1 — Two inventory examples
- **Activity:** Walk through inventory problem step by step; students define five elements
- **Key message:** This framework works for ANY sequential decision problem

### Topic 3: Pure Learning — Asset Selling (Week 3)
- **Content:** Optimal stopping; PFA (threshold) and VFA (value of waiting)
- **SDAM Reference:** Chapter 2
- **Activity:** Students implement threshold policy, vary the threshold, observe performance
- **First exposure** to policy design and evaluation

### Topic 4: Adaptive Planning with CFAs (Week 4)
- **Content:** Cost Function Approximations — parameterized optimization models
- **SDAM Reference:** Chapter 3
- **Activity:** Students parameterize a planning model and tune parameters
- **Key insight:** The method industry uses most, but academia teaches least

### Topic 5: Learning Under Uncertainty — Medical Decisions (Weeks 5-6)
- **Content:** Multi-armed bandits, Bayesian updating, belief states, exploration vs. exploitation
- **SDAM Reference:** Chapter 4 (Diabetes medication)
- **Policies covered:** Knowledge gradient, Thompson sampling, UCB, epsilon-greedy
- **Activity:** Students implement Thompson sampling for medication selection
- **Key insight:** Belief state B_t is part of the state variable

### Topic 6: Stochastic Networks (Week 7)
- **Content:** Static and dynamic shortest path problems
- **SDAM Reference:** Chapters 5-6
- **Policies covered:** VFA and DLA in network context
- **Activity:** Students implement shortest path with uncertain edge costs

### Topic 7: The Framework Revisited (Week 8) — CRITICAL SYNTHESIS
- **Content:** Revisit ALL five elements using examples from Topics 3-6
- **SDAM Reference:** Chapter 7
- **Covers:**
  - State variables including belief states
  - Scalar and vector-valued decisions
  - Different styles of modeling uncertainty
  - All four policy classes compared side-by-side
  - Different objective functions
- **Activity:** Students receive a NEW problem and must independently frame it using the five elements
- **This is the "aha moment"** — students see the universal pattern

### Topics 8-11: Advanced Applications (Weeks 9-14)
- **Energy storage** (SDAM Chs. 8-9) — ALL four policy classes illustrated
- **Supply chain** (SDAM Chs. 10-11) — Newsvendor, Beer Game
- **Ad-click optimization** (SDAM Ch. 12) — Online learning
- **Blood management** (SDAM Ch. 13) — Multi-dimensional resource allocation
- **Clinical trials** (SDAM Ch. 14) — Ethical learning
- **Choose based on audience** — see Section 5 below

---

## 3. Teaching Methods and Activities

### The Energy Storage "Aha Moment"
- **Setup:** Five variants of the SAME energy storage problem
- **Variant 1:** Simple price dynamics → PFA wins (buy low/sell high threshold)
- **Variant 2:** Complex price dynamics → CFA wins (parameterized optimization)
- **Variant 3:** Known value structure → VFA wins (approximate value function)
- **Variant 4:** Good forecast available → DLA wins (deterministic lookahead)
- **Variant 5:** Complex hybrid → Hybrid CFA+VFA wins
- **Takeaway:** NO single best method, even for essentially the same problem
- **Student reaction:** "I will never assume RL/DP is always the answer again"

### The Beer Game
- **Setup:** Classic multi-echelon supply chain simulation
- **Phase 1:** Students play the game manually → experience bullwhip effect
- **Phase 2:** Students model it as SDA (five elements)
- **Phase 3:** Students implement policies (PFA: base stock; CFA: parameterized ordering)
- **Phase 4:** Compare automated policies vs. their manual play
- **Takeaway:** Formal framework + systematic policies outperform human intuition

### Python Notebook Workflow
1. Students receive a Jupyter notebook with the model pre-implemented
2. They implement different policies (fill in the `get_decision` method)
3. They run simulations and compare policies
4. They tune parameters and observe sensitivity
5. They extend the model (add a new uncertainty source, change the objective)

### Policy Comparison Assignments
**Standard assignment structure:**
1. Given a new problem description (narrative)
2. Students frame it: define all five elements formally
3. Implement at least TWO different policy classes
4. Run Monte Carlo simulation to evaluate each
5. Write a 2-page analysis: when does each work best? why?
6. Present to class (5 minutes)

### Three-Phase Modeling Exercise
- Give students a real-world scenario (e.g., hospital nurse scheduling)
- Phase I: Students identify decisions, uncertainties, objectives (problem framing)
- Phase II: Students write down the five elements mathematically
- Phase III: Students implement and evaluate in Python

---

## 4. Common Student Misconceptions

### Misconception 1: "Deep RL is always the best approach"
- **Correction:** Show the energy storage example where a simple PFA or CFA outperforms deep RL
- **Explanation:** Deep RL (VFA with neural nets) is powerful but: requires massive data, can be unstable, hard to interpret, and often unnecessary when simpler policies work
- **Exercise:** Take a standard RL benchmark (CartPole). Implement a simple PFA. Show it works.

### Misconception 2: "You need Bellman's equation for sequential decisions"
- **Correction:** Bellman's equation is ONE approach (VFA class). PFA, CFA, and DLA don't use it at all
- **Powell's quote:** "Bellman's equation is the least useful of the four classes of policies"
- **Exercise:** Solve an inventory problem with CFA (no Bellman anywhere)

### Misconception 3: "More complex policies are always better"
- **Correction:** Energy storage example — simple threshold PFA beats complex VFA on some variants
- **Explanation:** Model complexity must match problem structure, not researcher sophistication
- **Exercise:** Compare a 2-parameter PFA vs. a neural network VFA on a simple problem

### Misconception 4: "State = what you can observe"
- **Correction:** State includes BELIEF state — what you think about unknowns
- **Explanation:** Without belief state, the problem is non-Markov (history-dependent)
- **Exercise:** Implement a bandit problem with and without tracking beliefs

### Misconception 5: "The model IS the solution"
- **Correction:** The model (five elements) defines the problem. The POLICY is the solution.
- **Explanation:** "Model first, then solve" — you can apply different policies to the same model
- **Exercise:** Same model, four different policies, compare results

### Misconception 6: "Dynamic programming = THE solution method"
- **Correction:** DP is one policy class (VFA). It's a way to compute a particular type of policy.
- **Exercise:** Classify 10 real-world decision methods into the four policy classes

### Misconception 7: "Stochastic programming = correct, deterministic = wrong"
- **Correction:** Parameterized deterministic models (CFA) can outperform stochastic programming
- **Explanation:** CFAs leverage the structure of deterministic models while handling uncertainty through parameter tuning
- **Exercise:** Compare a scenario-tree approach vs. CFA on an inventory problem

---

## 5. Adapting the Course to Different Audiences

### For Computer Science / AI Students
| Emphasis | De-emphasis |
|----------|-------------|
| VFA = Q-learning, DQN, actor-critic | Mathematical proofs |
| PFA = simple baselines that often win | Stochastic programming notation |
| MCTS = hybrid of all four classes | OR-specific applications |
| Connection to OpenAI Gym / Gymnasium | |
| Deep RL as one tool, not the only tool | |

**Key examples:** Ad-click optimization, game playing, recommendation systems, robotics
**Homework:** Take an RL benchmark → implement all four policy classes → compare

### For Business / MBA Students
| Emphasis | De-emphasis |
|----------|-------------|
| CFA (connects to their deterministic models) | Mathematical formalism |
| PFA (simple rules they already use) | Bellman equations |
| Supply chain (Beer Game, Newsvendor) | Algorithmic details |
| Energy and pricing applications | Convergence proofs |
| Industry practice vs. academic methods | |

**Key examples:** Inventory management, pricing, staffing, portfolio allocation
**Homework:** Identify a decision process in your company → frame as SDA → suggest policies

### For Engineering Students
| Emphasis | De-emphasis |
|----------|-------------|
| DLA = Model Predictive Control (MPC) | Business applications |
| Connection to optimal control theory | Bandit problems |
| Energy systems, robotics, manufacturing | Ad-click optimization |
| State-space models and transitions | |

**Key examples:** Energy storage, water reservoir management, process control, robotics
**Homework:** Take an MPC problem → show it's a DLA → compare with PFA or CFA

### For Data Science / Analytics Students
| Emphasis | De-emphasis |
|----------|-------------|
| A/B testing as multi-armed bandit SDA | Heavy optimization theory |
| Recommendation engines as learning SDA | Exact dynamic programming |
| Experimental design and adaptive sampling | Control theory |
| Connection to ML pipeline | |

**Key examples:** A/B testing, recommendation engines, experimental design, causal inference
**Homework:** Frame a real A/B test as SDA → implement Thompson sampling → compare with fixed allocation

### For Operations Research Students
| Emphasis | De-emphasis |
|----------|-------------|
| CFA as the "missing class" in OR education | RL terminology |
| Unification of DP, SP, SimOpt under SDA | Deep learning methods |
| How industry actually solves problems | Game-playing examples |
| Mathematical rigor of the framework | |

**Key examples:** Fleet management, scheduling, facility location, logistics
**Homework:** Take a classic OR paper → classify its method as one of the four classes → suggest alternatives

---

## 6. Assessment Strategies

### Formative Assessment (During Learning)
- **Quick-frame exercises:** Give a 2-sentence scenario, students identify the five elements in 5 minutes
- **Policy classification:** Given a real-world decision method, students classify it as PFA/CFA/VFA/DLA
- **Think-pair-share:** Two students discuss which policy class to try for a given problem

### Summative Assessment (Evaluating Learning)

**Exam Question Types:**
1. **Modeling question:** "A ride-sharing company must decide how to price rides and position drivers. Frame this as a sequential decision problem." (Define all five elements)
2. **Policy selection:** "Given this energy management problem, which policy class would you try first? Justify." (Must consider all four)
3. **Comparative analysis:** "Compare PFA and VFA for this inventory problem. Under what conditions would each be preferred?"
4. **Novel application:** "Identify a sequential decision problem in [healthcare / agriculture / education] that has not been studied. Frame it using the universal framework."

**Project Types:**
1. **Implementation project:** Complete SDA model + at least 2 policies + simulation + analysis
2. **Case study:** Analyze a real company's decision process through the SDA lens
3. **Literature review:** Survey papers from 2 different communities solving the same problem type → show how SDA unifies them
4. **Novel application:** Propose and model a new SDA application

---

## 7. Resources for Instructors

### Free Materials
| Resource | URL | Description |
|----------|-----|-------------|
| SDAM book (free PDF) | castle.princeton.edu/sdamodeling/ | Full undergraduate textbook |
| RLSO lecture notes | castle.princeton.edu/rlso/ | Graduate-level materials |
| Python modules | github.com/wbpowell328/stochastic-optimization | Original code |
| Refactored library | github.com/djanka2/stochastic-optimization | Modern SDPModel/SDPPolicy architecture |
| Community notebooks | github.com/Peymankor/seqdec_powell_repo | Chapter-by-chapter Jupyter notebooks |
| Lecture slides (Part I) | castle.princeton.edu/sda/ | Undergraduate slides |
| Lecture slides (Part II) | castle.princeton.edu/sda/ | Advanced slides |
| Supplementary materials | castlelab.princeton.edu/sdamodelingsupplements/ | Additional exercises |
| Teaching optimization monograph | castle.princeton.edu | 11-topic course outline |

### Workshop Materials
- 2-day workshop syllabus (Olin Business School, Washington University, Nov 2019)
- Tutorial materials for conference presentations
- Available from CASTLE Lab

---

## 8. Creating New Case Studies

### Step-by-Step Guide

**Step 1: Choose a Domain Relevant to Your Students**
- Ask: What industries do my students work in or aspire to?
- The same mathematical structure can wear many domain costumes

**Step 2: Find a Real Sequential Decision Process**
- Look for: repeated decisions, uncertainty between decisions, consequential outcomes
- Talk to practitioners: "What's the hardest recurring decision you face?"

**Step 3: Simplify to Capture Essential Structure**
- Remove details that don't affect the decision structure
- Aim for: 1-3 state variables, 1-2 decision variables, 1-2 sources of uncertainty
- Can add complexity later as extensions

**Step 4: Define the Five Elements**
- Write them out formally with clear notation
- This becomes the model specification

**Step 5: Implement Multiple Policy Classes**
- Implement at least 2 (ideally 3-4) policy classes
- Start with PFA (simplest), then CFA, then VFA or DLA
- Use the SDPModel/SDPPolicy architecture

**Step 6: Create a Jupyter Notebook**
- Section 1: Problem narrative and motivation
- Section 2: Model definition (five elements, with code)
- Section 3: Policy implementations
- Section 4: Simulation and comparison
- Section 5: Exercises for students

**Step 7: Design Exercises**
- Parameter tuning: "Find the best threshold for the PFA"
- Policy comparison: "Which policy class works best? Why?"
- Extension: "Add a new source of uncertainty and re-evaluate"
- Open-ended: "Propose a modification that would change which policy class is best"

---

## 9. The SDA Teaching Community

### Key Contributors
- **Warren B. Powell** (Princeton, emeritus) — Framework creator, author of SDAM and RLSO
- **Dennis Djanka** (Karlsruhe University, Germany) — Refactoring Python library, teaching SDA in Europe
- **Peymankor** — Community notebooks making SDAM accessible
- **Optimal Dynamics** — Industry application demonstrating SDA in trucking/logistics

### Growing Adoption
- SDA is being taught at universities in US, Germany, and expanding internationally
- Powell is developing his first MOOC on SDA
- Tutorial workshops at operations research and AI conferences
- The framework is gaining traction as a bridge between RL and OR communities

---

## 10. Frequently Asked Questions from Instructors

**Q: Do I need to cover all four policy classes?**
A: YES. This is the central pedagogical point. Covering only VFA (as in most RL courses) or only DLA (as in most MPC courses) gives students an incomplete toolkit.

**Q: How much math do students need?**
A: Basic probability (expectations, distributions) and basic optimization (what it means to minimize/maximize). No measure theory, no functional analysis, no advanced probability.

**Q: Can I teach this in one semester?**
A: Yes. The 11-topic outline fits a standard semester. Cover Topics 1-7 in the first half, then select 3-4 applications from Topics 8-11 based on your audience.

**Q: What programming background do students need?**
A: Basic Python. The notebooks are designed to be accessible. Students implement policies (short functions) rather than building infrastructure.

**Q: How does this relate to existing RL courses?**
A: SDA subsumes RL. RL focuses primarily on VFA (value function methods). SDA teaches all four classes and positions RL methods within the broader landscape. Ideal: offer SDA alongside or before an RL course.

**Q: Can I use this for a workshop instead of a full course?**
A: Yes. Powell has given 2-day workshops covering the framework and key applications. Focus on Topics 1-2, 5, 7, and the energy storage example.

---

## 11. Citations

- Powell, W.B. (2022). *Sequential Decision Analytics and Modeling: Modeling with Python*. Foundations and Trends in Technology, Information and Operations Management, 16(1-2), 1-176. NOW Publishers.
- Powell, W.B. (2024). *A Modern Approach to Teaching an Introduction to Optimization*. NOW Publishers.
- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions*. Wiley.
- Powell, W.B. (2019). "A unified framework for stochastic optimization." *European Journal of Operational Research*, 275(3), 795-821.
- CASTLE Lab, Princeton University. [castle.princeton.edu](https://castle.princeton.edu/)
- SDAM Teaching Materials. [castle.princeton.edu/sda/](https://castle.princeton.edu/sda/)
