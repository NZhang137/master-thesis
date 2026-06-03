# Master Thesis

**Working title:** Preference-Aware Coefficient Correction for Rewarded-Soups-Style Model Merging

## Goal

The main goal of this thesis is to study mappings of the form

$$
\boldsymbol{\lambda} = f(\mathbf{p}, \mathbf{R})
$$

where $\mathbf{p}$ is a user preference vector over objectives, $\mathbf{R}$ is a relationship matrix derived from task-vector or LoRA-adapter geometry, and $\boldsymbol{\lambda}$ are corrected merge coefficients for Rewarded-Soups-style model merging.

The thesis investigates whether merge coefficients can be selected more effectively than by the direct baseline

$$
\boldsymbol{\lambda} = \mathbf{p}.
$$

The focus is not on expanding the global Pareto front, but on improving coefficient selection within a fixed Rewarded-Soups-style interpolation family.

## Current Prototype

The current prototype is a small-scale Google Colab setup using:

- GPT-2 as a lightweight base model
- LoRA / PEFT for efficient adapter training
- Anthropic HH-RLHF as the first dataset
- separate helpfulness and harmlessness adapters as the first two-objective setup
- Rewarded-Soups-style adapter merging with different $\boldsymbol{\lambda}$ values

## Planned Extensions

After the initial GPT-2 prototype, the project may move to a more realistic small LLM setup using TinyLlama and multi-objective alignment datasets such as UltraFeedback or HelpSteer.

## Repository Structure

- `thesis/`: Proposal & LaTeX thesis draft
- `notebooks/`: Colab notebooks and experiments
- `src/`: reusable Python code
- `results/`: experiment outputs, tables, and plots
- `meetings/`: meeting slides

