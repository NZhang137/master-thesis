# HelpSteer2 All-Method Analysis Report

This report summarizes the full HelpSteer2 all-method merge evaluation. The comparison is inside the fixed Rewarded-Soups-style interpolation family.

The scores are lightweight proxy scores. They are not HelpSteer2 human labels, not reward-model scores, and do not establish global Pareto-front improvement.

## Main Winners

| preference_name     | method   | method_family   | hyperparameter_id   | hyperparameters_json     |   mean_utility |   improvement_over_direct_preference |   improvement_over_uniform |   l1_distance_to_p |   l2_distance_to_p |
|:--------------------|:---------|:----------------|:--------------------|:-------------------------|---------------:|-------------------------------------:|---------------------------:|-------------------:|-------------------:|
| balanced            | C2       | CAGrad-inspired | rho_1__tau_0p2      | {"rho": 1.0, "tau": 0.2} |       0.203964 |                             0.049563 |                   0.045137 |          0.0632049 |           0.037208 |
| detailed_answer     | C2       | CAGrad-inspired | rho_0p1__tau_0p1    | {"rho": 0.1, "tau": 0.1} |       0.203445 |                             0.045907 |                   0.06036  |          0.283397  |           0.159666 |
| helpfulness_focused | M2       | MGDA-inspired   | rho_0p1             | {"rho": 0.1}             |       0.361037 |                             0.130107 |                   0.170707 |          1         |           0.559055 |
| quality_focused     | M2       | MGDA-inspired   | rho_0p1             | {"rho": 0.1}             |       0.222406 |                             0.024459 |                   0.043832 |          0.660288  |           0.360353 |

## Mean Utility by Method

| method            |   mean_utility |
|:------------------|---------------:|
| P1                |       0.199467 |
| M2                |       0.195386 |
| P2                |       0.192874 |
| C1                |       0.18584  |
| C2                |       0.185679 |
| M1                |       0.185263 |
| direct_preference |       0.185204 |
| uniform           |       0.167704 |

## Comparison to Direct Preference and Uniform

| preference_name     | method   | hyperparameter_id   |   mean_utility |   improvement_over_direct_preference |   improvement_over_uniform |
|:--------------------|:---------|:--------------------|---------------:|-------------------------------------:|---------------------------:|
| balanced            | C2       | rho_1__tau_0p2      |       0.203964 |                             0.049563 |                   0.045137 |
| detailed_answer     | C2       | rho_0p1__tau_0p1    |       0.203445 |                             0.045907 |                   0.06036  |
| helpfulness_focused | M2       | rho_0p1             |       0.361037 |                             0.130107 |                   0.170707 |
| quality_focused     | M2       | rho_0p1             |       0.222406 |                             0.024459 |                   0.043832 |

## Utility and Distance to p

| preference_name     | method   | hyperparameter_id   |   mean_utility |   l1_distance_to_p |   l2_distance_to_p |
|:--------------------|:---------|:--------------------|---------------:|-------------------:|-------------------:|
| balanced            | C2       | rho_1__tau_0p2      |       0.203964 |          0.0632049 |          0.037208  |
| balanced            | C2       | rho_1__tau_0p05     |       0.196923 |          0.0815425 |          0.0399996 |
| balanced            | C1       | c_1__eps_1em08      |       0.185066 |          0.310314  |          0.15303   |
| balanced            | M2       | rho_0p1             |       0.184168 |          0.0509759 |          0.0234543 |
| balanced            | P1       | beta_1              |       0.184162 |          0         |          0         |
| detailed_answer     | C2       | rho_0p1__tau_0p1    |       0.203445 |          0.283397  |          0.159666  |
| detailed_answer     | C1       | c_0p5__eps_1em08    |       0.198604 |          0.16087   |          0.0745117 |
| detailed_answer     | P2       | eps_1em08__rho_1    |       0.188802 |          0         |          0         |
| detailed_answer     | C1       | c_0p25__eps_1em08   |       0.18355  |          0.16087   |          0.0745117 |
| detailed_answer     | M2       | rho_0p1             |       0.182118 |          0.241601  |          0.120696  |
| helpfulness_focused | M2       | rho_0p1             |       0.361037 |          1         |          0.559055  |
| helpfulness_focused | P2       | eps_1em08__rho_0p1  |       0.275246 |          0         |          0         |
| helpfulness_focused | P2       | eps_1em08__rho_10   |       0.271122 |          0         |          0         |
| helpfulness_focused | P1       | beta_2              |       0.263294 |          0         |          0         |
| helpfulness_focused | M1       | rho_10              |       0.263271 |          0.0325726 |          0.0153791 |
| quality_focused     | M2       | rho_0p1             |       0.222406 |          0.660288  |          0.360353  |
| quality_focused     | C1       | c_1__eps_1em08      |       0.222054 |          0.66087   |          0.341751  |
| quality_focused     | C2       | rho_0p1__tau_0p1    |       0.220295 |          0.319495  |          0.154731  |
| quality_focused     | P1       | beta_1              |       0.21878  |          0         |          0         |
| quality_focused     | P1       | beta_2              |       0.21848  |          0         |          0         |

## Computational Cost Observations

| method            |   mean_runtime_seconds |   max_runtime_seconds |   mean_peak_memory_mb |   solver_success_rate |
|:------------------|-----------------------:|----------------------:|----------------------:|----------------------:|
| C1                |            0.0320326   |           0.041237    |           0.0204145   |                     1 |
| C2                |            0.0414204   |           0.0631763   |           0.0165505   |                     1 |
| M1                |            0.143592    |           1.51732     |           1.77311     |                     1 |
| M2                |            0.0139222   |           0.0212562   |           0.0168273   |                     1 |
| P1                |            0.0002864   |           0.000431224 |           0.00584412  |                     1 |
| P2                |            0.00497863  |           0.00528566  |           0.0164388   |                     1 |
| direct_preference |            9.83195e-05 |           0.000152066 |           0.000884056 |                     1 |
| uniform           |            1.82995e-05 |           2.3871e-05  |           0.000291824 |                     1 |

## Limitations

- Proxy scores are deterministic heuristics, not HelpSteer2 labels.
- Generated responses do not automatically receive human attribute labels.
- The evaluation uses the fixed prompt set and tested hyperparameter grid only.
- No global Pareto-front improvement is claimed.

## Recommended Next Step

Inspect proxy utility together with distance to p and prompt-category behavior, then update the main HelpSteer2 prototype report with careful thesis-safe wording.
