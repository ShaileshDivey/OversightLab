# OversightLab — Learning Under Imperfect Oversight

**Exploratory research prototype | Operations management × AI safety**

OversightLab investigates how AI agents make resource-constrained operational decisions when oversight and feedback capture only part of the underlying objective. Its inventory-allocation environment distinguishes **economic decision quality**, **performance on selective inspection metrics**, and **execution validity**.

The central research question is whether feedback that improves measured performance also improves the underlying operation. The project is a proof of concept; it does **not** establish systematic reward hacking, strategic deception, or model-weight learning.

## Repository structure

```text
.
├── oversight_lab/     # Environments, agents, evaluation, analysis, and mock tools
├── tests/             # Tests for the allocation environment and evaluation protocols
├── README.md
└── requirements.txt
```

The research paper and full experimental outputs are **not included in this initial code release**. A draft manuscript will be added separately after revision. The repository is therefore not yet a self-contained reproduction package for the reported paid-model evaluations.

## Research approach

The main environment models the allocation of limited inventory across customers. Agents make decisions subject to resource constraints and receive operational feedback and, in selected conditions, inspection feedback. The evaluation framework separates:

1. **Economic quality:** value relative to the underlying allocation objective.
2. **Inspection performance:** score on the subset of outcomes monitored by the evaluator.
3. **Execution validity:** whether the agent returns a usable, protocol-compliant action.

The code supports sequential experiments, selective-oversight conditions, and controlled feedback-history evaluations. These experiments examine **in-context responses to information**, not reinforcement-learning updates to model weights.

## Install and test

Use Python from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Run a small mock allocation check without paid model calls:

```bash
python -m oversight_lab.allocation --model mock --rounds 2 --seeds 1 --output ./allocation_mock
```

Mock runs check basic software behavior; they do not reproduce results from paid AI-model evaluations. Full experiments may require provider access, API credentials, and separately saved experiment histories.

## Reproducibility and research status

- The public code supports inspection of the experimental logic and local testing.
- The current upload does not include the Colab notebook, working paper, raw trajectories, or complete result tables. These will be shared separately as appropriate.
- The experimental design includes both sequential decision trajectories and controlled, one-shot feedback-history comparisons. The two designs should not be treated as equivalent or pooled as independent replications.
- Strong conditional decision quality should not be conflated with reliable end-to-end execution.
- This project should not be interpreted as evidence that the tested agents intentionally optimized inspection scores at the expense of true economic value.

## Security

Never commit API keys, Colab secret values, private datasets, or provider authorization headers. If running paid-model experiments, supply secrets through a secure local environment or a supported secrets manager rather than hardcoding credentials.

## Paper

**Working paper: forthcoming in this repository after revision.** This repository is research code, not a claim of journal or conference publication or acceptance.
