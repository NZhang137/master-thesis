Working title: Preference-Aware Coefficient Correction for Rewarded-Soups-Style Model Merging

## Goal

The main goal is to study mappings of the form

$$
\boldsymbol{\lambda} = f(\mathbf{p}, \mathbf{R})
$$

where $\mathbf{p}$ is a user preference vector, $\mathbf{R}$ is a relationship matrix derived from task-vector or LoRA-adapter geometry, and $\boldsymbol{\lambda}$ are corrected merge coefficients for Rewarded-Soups-style model merging.

## Current Prototype

- GPT-2 + LoRA in Google Colab
- Anthropic HH-RLHF as first dataset
- Planned: train separate helpfulness and harmlessness adapters
- Planned: merge adapters with different \(\lambda\)-values
