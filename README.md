# Sequential Decision Analytics (SDA)

A comprehensive framework for **Sequential Decision Analytics and Reinforcement Learning**, based on Warren B. Powell's unified framework for sequential decisions under uncertainty.

## Overview

Sequential Decision Analytics (SDA) is a field centered on the broad problem class of **sequential decision problems** — any process involving the sequence: *decision, information, decision, information, ...* — drawing on a universal modeling framework and four meta-classes of policies.

This repository contains:

- **Claude Skills** — Reusable AI skills for reasoning about, teaching, framing, and solving sequential decision problems
- **Reference Materials** — Bibliography, resource guides, and curated links to key papers and books
- **Notebooks** — Jupyter notebooks with worked examples
- **Examples** — Python implementations of key SDA patterns
- **Presentations** — Slide decks and teaching materials

## The Universal Framework

Every sequential decision problem can be modeled using **five core elements**:

| Element | Symbol | Description |
|---------|--------|-------------|
| **State Variables** | S_t | All information needed to model the system from time t onward |
| **Decision Variables** | x_t | Actions/controls chosen at time t |
| **Exogenous Information** | W_t | New information arriving between decisions |
| **Transition Function** | S^M(S_t, x_t, W_{t+1}) | How state evolves over time |
| **Objective Function** | max E[sum C(S_t, x_t)] | Performance metric optimized over policies |

## The Four Classes of Policies

Any method for making decisions belongs to one of four meta-classes (or a hybrid):

1. **Policy Function Approximations (PFAs)** — Analytical rules mapping states to actions
2. **Cost Function Approximations (CFAs)** — Parameterized optimization models
3. **Value Function Approximations (VFAs)** — Approximate the downstream value of states
4. **Direct Lookahead Approximations (DLAs)** — Plan over a future horizon

## Claude Skills

Skills in `.claude/skills/` provide reusable knowledge for any Claude model:

| Skill | Purpose |
|-------|---------|
| `sda-universal-modeling-framework.md` | Core modeling framework and five elements |
| `sda-four-classes-of-policies.md` | Deep dive into PFA, CFA, VFA, DLA |
| `sda-problem-framing.md` | How to frame any problem as a sequential decision |
| `sda-application-domains.md` | Case studies across energy, health, supply chain, etc. |
| `sda-belief-states-and-learning.md` | Bayesian beliefs, exploration/exploitation, knowledge gradient |
| `sda-reasoning-and-discovery.md` | Applying SDA thinking to new problem domains |
| `sda-python-patterns.md` | Implementation patterns using SDPModel/SDPPolicy |
| `sda-teaching-and-pedagogy.md` | How to teach SDA effectively |

## Key References

- Powell, W.B. (2022). *Sequential Decision Analytics and Modeling: Modeling with Python*. NOW Publishers.
- Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions*. Wiley.
- Powell, W.B. (2019). "A unified framework for stochastic optimization." *European Journal of Operational Research*, 275(3), 795-821.
- Powell, W.B. & Meisel, S. (2016). "Tutorial on Stochastic Optimization in Energy Part II: An Energy Storage Illustration." *IEEE Trans. Power Systems*, 31(2), 1468-1475.

## External Resources

- [CASTLE Lab, Princeton University](https://castle.princeton.edu/)
- [SDA Teaching Materials](https://castle.princeton.edu/sda/)
- [SDA Modeling Book (free PDF)](https://castle.princeton.edu/sdamodeling/)
- [RLSO Book](https://castle.princeton.edu/rlso/)
- [Official Python Library](https://github.com/wbpowell328/stochastic-optimization)
- [Refactored Library (Djanka)](https://github.com/djanka2/stochastic-optimization)
- [Community Notebooks (Peymankor)](https://github.com/Peymankor/seqdec_powell_repo)
- [The Jungle of Stochastic Optimization](https://castle.princeton.edu/jungle/)

## License

Educational and research use. All intellectual credit for the SDA framework belongs to Warren B. Powell and collaborators.
