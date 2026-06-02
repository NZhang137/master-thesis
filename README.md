Working title: Preference-Aware Coefficient Correction for Rewarded-Soups-Style Model Merging

## Goal

This repository contains code, notes, experiments, and thesis drafts for my master's thesis.

The main goal is to study mappings of the form

\[
\lambda = f(p, R)
\]

where \(p\) is a user preference vector, \(R\) is a relationship matrix derived from task-vector or LoRA-adapter geometry, and \(\lambda\) are corrected merge coefficients for Rewarded-Soups-style model merging.

## Current Prototype

- GPT-2 + LoRA in Google Colab
- Anthropic HH-RLHF as first dataset
- Planned: train separate helpfulness and harmlessness adapters
- Planned: merge adapters with different \(\lambda\)-values
