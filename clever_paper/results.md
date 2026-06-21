# Unified Pipeline Results: Tests 1-9

- Series directory: `clever_paper\analysis\unified_test_series\series_20260621_121023`
- Manifest: `clever_paper\analysis\unified_test_series\series_20260621_121023\series_manifest.json`
- Author comparison data: `clever_paper\data\fig3\data_better.npz`
- Generated: `2026-06-21T11:01:54`
- Completed tests included: `01, 02, 03, 04, 05, 06, 07, 08, 09`

## Interpretation Note

Tests 1-4 are progression/smoke tests, test 6 is a high-temperature stress bracket, and tests 5/7/8/9 are the main writeup-grade comparison set. The reported `insertable` observable is the v7 causal insertion-decay observable; raw projection values are preserved beside it.

## Summary Table

| Test | Name | Pass | Version | Stage | L | q | n_traj | Temps | T_ins | Raw MAE | Insert MAE | Max Err | Bias | RMSE | Kicks | Hamming |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | q512_cuda_smoke | True | 2.8.0 | causal_insertion_decay_v7 | 8 | 512 | 16 | 5 | 0.840 | 0.34379 | 0.26008 | 0.55802 | 0.26008 | 0.33692 | 32 | 36 |
| 02 | q512_short | True | 2.8.0 | causal_insertion_decay_v7 | 8 | 512 | 16 | 5 | 0.720 | 0.31718 | 0.18494 | 0.35091 | -0.04332 | 0.21968 | 28 | 28 |
| 03 | q512_smoke_insertion_decay | True | 2.8.0 | causal_insertion_decay_v7 | 8 | 512 | 32 | 9 | 0.720 | 0.31016 | 0.08186 | 0.17434 | -0.07322 | 0.10192 | 29 | 35 |
| 04 | q512_mid_freeze | True | 2.8.0 | causal_insertion_decay_v7 | 8 | 512 | 64 | 9 | 0.800 | 0.28408 | 0.08046 | 0.20311 | 0.08046 | 0.10696 | 50 | 64 |
| 05 | q512_frozen_candidate | True | 2.8.0 | causal_insertion_decay_v7 | 8 | 512 | 64 | 9 | 0.840 | 0.27266 | 0.04709 | 0.11693 | 0.04090 | 0.06284 | 61 | 101 |
| 06 | high_temp_bracket | True | 2.8.0 | causal_insertion_decay_v7 | 8 | 512 | 64 | 7 | 0.840 | 0.37970 | 0.09458 | 0.27853 | 0.07527 | 0.13543 | 93 | 146 |
| 07 | q1024_continuum_check | True | 2.8.0 | causal_insertion_decay_v7 | 8 | 1024 | 64 | 9 | 0.760 | 0.28478 | 0.04028 | 0.08756 | 0.02329 | 0.04845 | 55 | 68 |
| 08 | L16_medium | True | 2.8.0 | causal_insertion_decay_v7 | 16 | 512 | 64 | 9 | 0.720 | 0.23897 | 0.04747 | 0.13302 | -0.04596 | 0.06409 | 93 | 106 |
| 09 | L32_deep | True | 2.8.0 | causal_insertion_decay_v7 | 32 | 512 | 100 | 9 | 0.760 | 0.19474 | 0.04180 | 0.10276 | -0.04163 | 0.05291 | 449 | 495 |

## Main Comparison Set

| Test | Name | T_ins | Guide T | Raw MAE | Insert MAE | Max Err | Bias | Lag Strength | Thermal Strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 05 | q512_frozen_candidate | 0.840 | 0.840 | 0.27266 | 0.04709 | 0.11693 | 0.04090 | 0.825 | 0.925 |
| 07 | q1024_continuum_check | 0.760 | 0.760 | 0.28478 | 0.04028 | 0.08756 | 0.02329 | 0.825 | 0.925 |
| 08 | L16_medium | 0.720 | 0.720 | 0.23897 | 0.04747 | 0.13302 | -0.04596 | 0.825 | 0.925 |
| 09 | L32_deep | 0.760 | 0.760 | 0.19474 | 0.04180 | 0.10276 | -0.04163 | 0.825 | 0.925 |

- Main-set insertable MAE range: `0.04028` to `0.04747`.
- Main-set raw MAE range: `0.19474` to `0.28478`.
- L32 holdout is test 09; it remains below `0.05` insertable MAE.

## Test 01: q512_cuda_smoke

### Raw Console Output

```text
[UNIFIED] T=0.4000 |Psi_g|=1.7908e-01 chi=4.3947e-02 pairUPS=9.714e+05
[UNIFIED] T=0.6400 |Psi_g|=1.1102e-01 chi=3.6867e-02 pairUPS=9.895e+05
[UNIFIED] T=0.7200 |Psi_g|=8.7957e-02 chi=6.1275e-02 pairUPS=9.880e+05
[UNIFIED] T=0.8400 |Psi_g|=8.9323e-02 chi=1.3424e-01 pairUPS=1.004e+06
[UNIFIED] T=0.9600 |Psi_g|=7.1744e-02 chi=6.5255e-02 pairUPS=9.850e+05
[UNIFIED] passed              = True
[UNIFIED] projection base     = analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\projection_base.npz
[UNIFIED] T_ins               = 0.840000
[UNIFIED] scout score         = 1.118815e-01
[UNIFIED] insertion decay     = 8.595121e-01
[UNIFIED] lambda0             = 9.493727e-01
[UNIFIED] K2 live kicks       = 32
[UNIFIED] live-post hamming   = 36
[UNIFIED] evidence            = analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_evidence_20260621_121023.json
```

### Console Summary

| Field | Value |
| --- | --- |
| passed | True |
| projection base | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\projection_base.npz |
| T_ins | 0.840000 |
| scout score | 0.111881 |
| insertion decay | 0.859512 |
| lambda0 | 0.949373 |
| K2 live kicks | 32 |
| live-post hamming | 36 |
| evidence | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_evidence_20260621_121023.json |
| log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\01_q512_cuda_smoke.log |

### Run Configuration

| Config | Value |
| --- | --- |
| L | 8 |
| q | 512 |
| n_traj | 16 |
| repeats | 1 |
| burn_steps | 30000 |
| sample_steps | 12000 |
| sample_stride | 1000 |
| checkpoint_steps | 3000 |
| forward_steps | 8000 |
| forward_sample_stride | 1000 |
| temps | 0.40,0.64,0.72,0.84,0.96 |
| Psi_scale_pi | 0.85 |
| proposal_id | 2 |
| proposal_delta_rad | 0.1 |
| window_radius | 1 |
| init | ordered |
| seed | 1234 |
| forward_seed | 987654 |
| threads_per_block | 128 |
| tap_lags | 1,2,3 |
| tap_weights | 0.5,0.3,0.2 |
| beta_pi | 0.5 |
| rho | 0.1 |
| lambda_safety_factor | 3.0 |
| null_scale | 0.49 |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | None |

### Core Metrics

| Metric | Value |
| --- | --- |
| raw_mae_vs_author | 0.34378645 |
| insertable_mae_vs_author | 0.26008464 |
| insertable_max_abs_error | 0.55802222 |
| insertable_bias | 0.26008464 |
| insertable_rmse | 0.33691785 |
| K1_max_abs_error | 4.16564866e-16 |
| K1_passed | True |
| selector_is_edge | False |
| selector_score | 0.11188149 |
| K2_live_total_kicks | 32 |
| K2_live_kicked_trajectories | 15 |
| K2_live_max_abs_kick_float | 1.50000000 |
| K2_null_total_kicks | 0 |
| live_post_hamming | 36 |

### Causal Insertion Decay

| Field | Value |
| --- | --- |
| stage | causal_insertion_decay_v7 |
| enabled | True |
| uses_author_data | False |
| decay_formula | fit log(coherence_lag)=a-rate*lag across causal taps; rate_excess=max(0, rate(T)-min_T rate(T)); lag_slope_decay=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints); thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max), with T_guide=min(T_ins, simulated chi shoulder); thermal_envelope_factor=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints*(thermal_gain(T)-1)); D_insert(T)=min(1, lag_slope_decay*thermal_envelope_factor); auto horizon=max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps; Psi_Y still uses coherence-thinned taps for K2 freeze; finite-clock insertion_threshold_survival is recorded as a K2 readiness diagnostic, not multiplied into the observable; c_mean_m_insertable=c_mean_m_raw*D_insert(T); lag_strength and default kappa are frozen global finite-horizon calibration constants |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | 0.925 |
| calibration_note | single frozen global finite-horizon calibration; no runtime author-data lookup |
| guide_center_T | 0.84 |
| guide_center_mode | min(T_ins, chi_shoulder) |
| chi_shoulder_T | 0.84 |
| guide_iteration | 1 |
| guide_fixed_point_converged | True |
| guide_width | 0.18666666666666665 |
| guide_strength_mode | frozen_global_default |
| horizon_mode | auto |
| horizon_checkpoints | 7.366666666666667 |
| rate_floor_rule | minimum fitted lag-coherence decay rate across the simulated temperature grid |
| threshold_role | diagnostic/readiness gate for K2 insertion, not an observable multiplier |

### Scout Selector Rows

| T | score | |Psi_g| | chi | c_mean_m | disorder | decay |
| --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.00000000 | 0.17907912 | 0.04394652 | 0.92244602 | 0.07755398 | 1.00000000 |
| 0.640 | 0.00000000 | 0.11102410 | 0.03686661 | 0.88698556 | 0.11301444 | 0.98062398 |
| 0.720 | 0.02238206 | 0.08795700 | 0.06127516 | 0.86752549 | 0.13247451 | 0.90988244 |
| 0.840 | 0.11188149 | 0.08932346 | 0.13424347 | 0.85526124 | 0.14473876 | 0.85951213 |
| 0.960 | 0.00000000 | 0.07174371 | 0.06525481 | 0.83791142 | 0.16208858 | 0.75774323 |

### Temperature Curve Rows

| T | |Psi_g| | chi | raw M | insertable M | author M | raw err | insert err | decay | lag decay | thermal env | guide gain | excess rate | pair UPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.179079 | 0.043947 | 0.922446 | 0.922446 | 0.900455 | 0.021991 | 0.021991 | 1.000000 | 1.000000 | 1.000000 | 0.250000 | 0.000000 | 9.714e+05 |
| 0.640 | 0.111024 | 0.036867 | 0.886986 | 0.869799 | 0.763384 | 0.123602 | 0.106416 | 0.980624 | 0.948652 | 1.033703 | 0.371179 | 0.008674 | 9.895e+05 |
| 0.720 | 0.087957 | 0.061275 | 0.867525 | 0.789346 | 0.649498 | 0.218028 | 0.139848 | 0.909882 | 0.842686 | 1.079741 | 0.551760 | 0.028163 | 9.880e+05 |
| 0.840 | 0.089323 | 0.134243 | 0.855261 | 0.735107 | 0.260962 | 0.594299 | 0.474145 | 0.859512 | 0.859512 | 1.000000 | 1.000000 | 0.024910 | 1.004e+06 |
| 0.960 | 0.071744 | 0.065255 | 0.837911 | 0.634922 | 0.076899 | 0.761012 | 0.558022 | 0.757743 | 0.858075 | 0.883074 | 1.812384 | 0.025185 | 9.850e+05 |

### Output Artifacts

| Artifact | Path |
| --- | --- |
| anchor_evidence_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_anchor_evidence_20260621_121023.csv |
| checkpoint_manifest_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_checkpoint_manifest_20260621_121023.csv |
| dynamic_trajectory_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_dynamic_trajectory_20260621_121023.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_evidence_20260621_121023.json |
| freeze_manifest_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_freeze_manifest_20260621_121023.json |
| live_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_live_insertion_rows_20260621_121023.csv |
| null_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_null_insertion_rows_20260621_121023.csv |
| out_dir | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023 |
| projection_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\projection_base.npz |
| smallN_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\base_smallN_L2_q8.npz |
| temp_curve_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_temp_curve_20260621_121023.csv |
| trajectory_summary_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_trajectory_summary_20260621_121023.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\unified_main_pipeline\20260621_121023\unified_evidence_20260621_121023.json |
| runner_log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\01_q512_cuda_smoke\01_q512_cuda_smoke.log |

### Freeze And Dynamic Outputs

| Field | Value |
| --- | --- |
| freeze_passed | True |
| Pi_lambda0 | 0.9493727388678912 |
| Pi_anchor_count | None |
| Pi_candidate_count | None |
| causal_tap_lags | None |
| causal_tap_weights | None |
| resolution_gate_passed | True |
| dynamic_gates | {"dynamic_diverged": true, "live_post_state_hamming_final_total": 36, "max_abs_live_minus_post_final": 0.0014982978112845213, "mean_live_minus_post_final": -9.408613294970697e-05, "null_matches_post_final": true, "null_post_state_hamming_final_total": 0} |
| live_summary | {"kicked_trajectories": 15, "lambda0": 0.9493727388678912, "max_abs_kick_float": 1.5, "total_kicks": 32} |
| null_summary | {"kicked_trajectories": 0, "lambda0": 0.15506421401508888, "max_abs_kick_float": 0.24499999999999997, "total_kicks": 0} |
| live_post_hamming | 36 |

## Test 02: q512_short

### Raw Console Output

```text
[UNIFIED] T=0.4000 |Psi_g|=2.6448e-01 chi=1.0658e-01 pairUPS=9.851e+05
[UNIFIED] T=0.6400 |Psi_g|=1.8154e-01 chi=7.9088e-02 pairUPS=9.992e+05
[UNIFIED] T=0.7200 |Psi_g|=1.3073e-01 chi=1.6274e-01 pairUPS=9.770e+05
[UNIFIED] T=0.8400 |Psi_g|=1.1881e-01 chi=1.5934e-01 pairUPS=1.011e+06
[UNIFIED] T=0.9600 |Psi_g|=1.0616e-01 chi=1.6306e-01 pairUPS=1.037e+06
[UNIFIED] passed              = True
[UNIFIED] projection base     = analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\projection_base.npz
[UNIFIED] T_ins               = 0.720000
[UNIFIED] scout score         = 5.360018e-02
[UNIFIED] insertion decay     = 5.116227e-01
[UNIFIED] lambda0             = 5.989612e-01
[UNIFIED] K2 live kicks       = 28
[UNIFIED] live-post hamming   = 28
[UNIFIED] evidence            = analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_evidence_20260621_121035.json
```

### Console Summary

| Field | Value |
| --- | --- |
| passed | True |
| projection base | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\projection_base.npz |
| T_ins | 0.720000 |
| scout score | 0.053600 |
| insertion decay | 0.511623 |
| lambda0 | 0.598961 |
| K2 live kicks | 28 |
| live-post hamming | 28 |
| evidence | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_evidence_20260621_121035.json |
| log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\02_q512_short.log |

### Run Configuration

| Config | Value |
| --- | --- |
| L | 8 |
| q | 512 |
| n_traj | 16 |
| repeats | 1 |
| burn_steps | 60000 |
| sample_steps | 22000 |
| sample_stride | 1000 |
| checkpoint_steps | 5500 |
| forward_steps | 10000 |
| forward_sample_stride | 1000 |
| temps | 0.40,0.64,0.72,0.84,0.96 |
| Psi_scale_pi | 0.85 |
| proposal_id | 2 |
| proposal_delta_rad | 0.1 |
| window_radius | 1 |
| init | ordered |
| seed | 1234 |
| forward_seed | 987654 |
| threads_per_block | 128 |
| tap_lags | 1,2,3 |
| tap_weights | 0.5,0.3,0.2 |
| beta_pi | 0.5 |
| rho | 0.1 |
| lambda_safety_factor | 3.0 |
| null_scale | 0.49 |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | None |

### Core Metrics

| Metric | Value |
| --- | --- |
| raw_mae_vs_author | 0.31717545 |
| insertable_mae_vs_author | 0.18493539 |
| insertable_max_abs_error | 0.35091096 |
| insertable_bias | -0.04332228 |
| insertable_rmse | 0.21967607 |
| K1_max_abs_error | 3.19490668e-16 |
| K1_passed | True |
| selector_is_edge | False |
| selector_score | 0.05360018 |
| K2_live_total_kicks | 28 |
| K2_live_kicked_trajectories | 14 |
| K2_live_max_abs_kick_float | 1.50000000 |
| K2_null_total_kicks | 0 |
| live_post_hamming | 28 |

### Causal Insertion Decay

| Field | Value |
| --- | --- |
| stage | causal_insertion_decay_v7 |
| enabled | True |
| uses_author_data | False |
| decay_formula | fit log(coherence_lag)=a-rate*lag across causal taps; rate_excess=max(0, rate(T)-min_T rate(T)); lag_slope_decay=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints); thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max), with T_guide=min(T_ins, simulated chi shoulder); thermal_envelope_factor=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints*(thermal_gain(T)-1)); D_insert(T)=min(1, lag_slope_decay*thermal_envelope_factor); auto horizon=max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps; Psi_Y still uses coherence-thinned taps for K2 freeze; finite-clock insertion_threshold_survival is recorded as a K2 readiness diagnostic, not multiplied into the observable; c_mean_m_insertable=c_mean_m_raw*D_insert(T); lag_strength and default kappa are frozen global finite-horizon calibration constants |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | 0.925 |
| calibration_note | single frozen global finite-horizon calibration; no runtime author-data lookup |
| guide_center_T | 0.4 |
| guide_center_mode | min(T_ins, chi_shoulder) |
| chi_shoulder_T | 0.4 |
| guide_iteration | 1 |
| guide_fixed_point_converged | True |
| guide_width | 0.18666666666666665 |
| guide_strength_mode | frozen_global_default |
| horizon_mode | auto |
| horizon_checkpoints | 6.5181818181818185 |
| rate_floor_rule | minimum fitted lag-coherence decay rate across the simulated temperature grid |
| threshold_role | diagnostic/readiness gate for K2 insertion, not an observable multiplier |

### Scout Selector Rows

| T | score | |Psi_g| | chi | c_mean_m | disorder | decay |
| --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.00000000 | 0.26447686 | 0.10657690 | 0.90859123 | 0.09140877 | 1.00000000 |
| 0.640 | 0.00000000 | 0.18153925 | 0.07908845 | 0.86293311 | 0.13706689 | 0.47798912 |
| 0.720 | 0.05360018 | 0.13072837 | 0.16273627 | 0.84000304 | 0.15999696 | 0.51162272 |
| 0.840 | 0.03028613 | 0.11880716 | 0.15933655 | 0.81817341 | 0.18182659 | 0.44400811 |
| 0.960 | 0.00000000 | 0.10615789 | 0.16305639 | 0.80737392 | 0.19262608 | 0.39694361 |

### Temperature Curve Rows

| T | |Psi_g| | chi | raw M | insertable M | author M | raw err | insert err | decay | lag decay | thermal env | guide gain | excess rate | pair UPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.264477 | 0.106577 | 0.908591 | 0.908591 | 0.900455 | 0.008137 | 0.008137 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 9.851e+05 |
| 0.640 | 0.181539 | 0.079088 | 0.862933 | 0.412473 | 0.763384 | 0.099550 | -0.350911 | 0.477989 | 0.798735 | 0.598433 | 3.284734 | 0.041790 | 9.992e+05 |
| 0.720 | 0.130728 | 0.162736 | 0.840003 | 0.429765 | 0.649498 | 0.190505 | -0.219733 | 0.511623 | 0.845741 | 0.604940 | 4.000000 | 0.031156 | 9.770e+05 |
| 0.840 | 0.118807 | 0.159337 | 0.818173 | 0.363276 | 0.260962 | 0.557211 | 0.102314 | 0.444008 | 0.816296 | 0.543930 | 4.000000 | 0.037746 | 1.011e+06 |
| 0.960 | 0.106158 | 0.163056 | 0.807374 | 0.320482 | 0.076899 | 0.730474 | 0.243582 | 0.396944 | 0.793747 | 0.500088 | 4.000000 | 0.042955 | 1.037e+06 |

### Output Artifacts

| Artifact | Path |
| --- | --- |
| anchor_evidence_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_anchor_evidence_20260621_121035.csv |
| checkpoint_manifest_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_checkpoint_manifest_20260621_121035.csv |
| dynamic_trajectory_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_dynamic_trajectory_20260621_121035.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_evidence_20260621_121035.json |
| freeze_manifest_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_freeze_manifest_20260621_121035.json |
| live_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_live_insertion_rows_20260621_121035.csv |
| null_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_null_insertion_rows_20260621_121035.csv |
| out_dir | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035 |
| projection_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\projection_base.npz |
| smallN_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\base_smallN_L2_q8.npz |
| temp_curve_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_temp_curve_20260621_121035.csv |
| trajectory_summary_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_trajectory_summary_20260621_121035.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\unified_main_pipeline\20260621_121035\unified_evidence_20260621_121035.json |
| runner_log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\02_q512_short\02_q512_short.log |

### Freeze And Dynamic Outputs

| Field | Value |
| --- | --- |
| freeze_passed | True |
| Pi_lambda0 | 0.598961194582668 |
| Pi_anchor_count | None |
| Pi_candidate_count | None |
| causal_tap_lags | None |
| causal_tap_weights | None |
| resolution_gate_passed | True |
| dynamic_gates | {"dynamic_diverged": true, "live_post_state_hamming_final_total": 28, "max_abs_live_minus_post_final": 0.0003953406892185374, "mean_live_minus_post_final": 1.7642751711025029e-06, "null_matches_post_final": true, "null_post_state_hamming_final_total": 0} |
| live_summary | {"kicked_trajectories": 14, "lambda0": 0.598961194582668, "max_abs_kick_float": 1.4999999999999998, "total_kicks": 28} |
| null_summary | {"kicked_trajectories": 0, "lambda0": 0.09783032844850245, "max_abs_kick_float": 0.245, "total_kicks": 0} |
| live_post_hamming | 28 |

## Test 03: q512_smoke_insertion_decay

### Raw Console Output

```text
[UNIFIED] T=0.4000 |Psi_g|=3.8379e-01 chi=6.3816e-02 pairUPS=1.961e+06
[UNIFIED] T=0.5600 |Psi_g|=3.1170e-01 chi=9.6936e-02 pairUPS=1.975e+06
[UNIFIED] T=0.6400 |Psi_g|=2.0719e-01 chi=1.9733e-01 pairUPS=1.980e+06
[UNIFIED] T=0.7200 |Psi_g|=1.8598e-01 chi=2.4842e-01 pairUPS=1.978e+06
[UNIFIED] T=0.7600 |Psi_g|=2.2874e-01 chi=1.2253e-01 pairUPS=1.994e+06
[UNIFIED] T=0.8000 |Psi_g|=1.6498e-01 chi=2.3144e-01 pairUPS=1.992e+06
[UNIFIED] T=0.8400 |Psi_g|=1.3527e-01 chi=2.4476e-01 pairUPS=2.003e+06
[UNIFIED] T=0.8800 |Psi_g|=1.2358e-01 chi=2.1458e-01 pairUPS=2.005e+06
[UNIFIED] T=0.9600 |Psi_g|=1.2943e-01 chi=3.3375e-01 pairUPS=2.016e+06
[UNIFIED] passed              = True
[UNIFIED] projection base     = analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\projection_base.npz
[UNIFIED] T_ins               = 0.720000
[UNIFIED] scout score         = 4.864260e-02
[UNIFIED] insertion decay     = 5.805524e-01
[UNIFIED] lambda0             = 2.646687e-01
[UNIFIED] K2 live kicks       = 29
[UNIFIED] live-post hamming   = 35
[UNIFIED] evidence            = analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_evidence_20260621_121054.json
```

### Console Summary

| Field | Value |
| --- | --- |
| passed | True |
| projection base | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\projection_base.npz |
| T_ins | 0.720000 |
| scout score | 0.048643 |
| insertion decay | 0.580552 |
| lambda0 | 0.264669 |
| K2 live kicks | 29 |
| live-post hamming | 35 |
| evidence | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_evidence_20260621_121054.json |
| log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\03_q512_smoke_insertion_decay.log |

### Run Configuration

| Config | Value |
| --- | --- |
| L | 8 |
| q | 512 |
| n_traj | 32 |
| repeats | 1 |
| burn_steps | 120000 |
| sample_steps | 44000 |
| sample_stride | 2000 |
| checkpoint_steps | 11000 |
| forward_steps | 20000 |
| forward_sample_stride | 1000 |
| temps | 0.40,0.56,0.64,0.72,0.76,0.80,0.84,0.88,0.96 |
| Psi_scale_pi | 0.85 |
| proposal_id | 2 |
| proposal_delta_rad | 0.1 |
| window_radius | 1 |
| init | ordered |
| seed | 1234 |
| forward_seed | 987654 |
| threads_per_block | 128 |
| tap_lags | 1,2,3 |
| tap_weights | 0.5,0.3,0.2 |
| beta_pi | 0.5 |
| rho | 0.1 |
| lambda_safety_factor | 3.0 |
| null_scale | 0.49 |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | None |

### Core Metrics

| Metric | Value |
| --- | --- |
| raw_mae_vs_author | 0.31016181 |
| insertable_mae_vs_author | 0.08185563 |
| insertable_max_abs_error | 0.17433714 |
| insertable_bias | -0.07322147 |
| insertable_rmse | 0.10192482 |
| K1_max_abs_error | 4.16333634e-16 |
| K1_passed | True |
| selector_is_edge | False |
| selector_score | 0.04864260 |
| K2_live_total_kicks | 29 |
| K2_live_kicked_trajectories | 11 |
| K2_live_max_abs_kick_float | 1.50000000 |
| K2_null_total_kicks | 0 |
| live_post_hamming | 35 |

### Causal Insertion Decay

| Field | Value |
| --- | --- |
| stage | causal_insertion_decay_v7 |
| enabled | True |
| uses_author_data | False |
| decay_formula | fit log(coherence_lag)=a-rate*lag across causal taps; rate_excess=max(0, rate(T)-min_T rate(T)); lag_slope_decay=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints); thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max), with T_guide=min(T_ins, simulated chi shoulder); thermal_envelope_factor=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints*(thermal_gain(T)-1)); D_insert(T)=min(1, lag_slope_decay*thermal_envelope_factor); auto horizon=max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps; Psi_Y still uses coherence-thinned taps for K2 freeze; finite-clock insertion_threshold_survival is recorded as a K2 readiness diagnostic, not multiplied into the observable; c_mean_m_insertable=c_mean_m_raw*D_insert(T); lag_strength and default kappa are frozen global finite-horizon calibration constants |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | 0.925 |
| calibration_note | single frozen global finite-horizon calibration; no runtime author-data lookup |
| guide_center_T | 0.64 |
| guide_center_mode | min(T_ins, chi_shoulder) |
| chi_shoulder_T | 0.64 |
| guide_iteration | 1 |
| guide_fixed_point_converged | True |
| guide_width | 0.18666666666666665 |
| guide_strength_mode | frozen_global_default |
| horizon_mode | auto |
| horizon_checkpoints | 6.5181818181818185 |
| rate_floor_rule | minimum fitted lag-coherence decay rate across the simulated temperature grid |
| threshold_role | diagnostic/readiness gate for K2 insertion, not an observable multiplier |

### Scout Selector Rows

| T | score | |Psi_g| | chi | c_mean_m | disorder | decay |
| --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.00000000 | 0.38378560 | 0.06381634 | 0.91500304 | 0.08499696 | 1.00000000 |
| 0.560 | 0.01797197 | 0.31169963 | 0.09693585 | 0.87215290 | 0.12784710 | 0.89354617 |
| 0.640 | 0.04428422 | 0.20719393 | 0.19732697 | 0.84277063 | 0.15722937 | 0.72896022 |
| 0.720 | 0.04864260 | 0.18598150 | 0.24842343 | 0.81846309 | 0.18153691 | 0.58055241 |
| 0.760 | 0.02398785 | 0.22874096 | 0.12252615 | 0.81662000 | 0.18338000 | 0.52419262 |
| 0.800 | 0.01752098 | 0.16498130 | 0.23143875 | 0.80157087 | 0.19842913 | 0.29541925 |
| 0.840 | 0.00672241 | 0.13526623 | 0.24476055 | 0.78049375 | 0.21950625 | 0.31363385 |
| 0.880 | 0.00000000 | 0.12357649 | 0.21457619 | 0.78152010 | 0.21847990 | 0.14009560 |
| 0.960 | 0.00313657 | 0.12943148 | 0.33374994 | 0.72602115 | 0.27397885 | 0.13939637 |

### Temperature Curve Rows

| T | |Psi_g| | chi | raw M | insertable M | author M | raw err | insert err | decay | lag decay | thermal env | guide gain | excess rate | pair UPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.383786 | 0.063816 | 0.915003 | 0.915003 | 0.900455 | 0.014548 | 0.014548 | 1.000000 | 1.000000 | 1.000000 | 0.304439 | 0.000000 | 1.961e+06 |
| 0.560 | 0.311700 | 0.096936 | 0.872153 | 0.779309 | 0.827175 | 0.044978 | -0.047866 | 0.893546 | 0.845931 | 1.056287 | 0.672718 | 0.031114 | 1.975e+06 |
| 0.640 | 0.207194 | 0.197327 | 0.842771 | 0.614346 | 0.763384 | 0.079387 | -0.149037 | 0.728960 | 0.728960 | 1.000000 | 1.000000 | 0.058789 | 1.980e+06 |
| 0.720 | 0.185982 | 0.248423 | 0.818463 | 0.475161 | 0.649498 | 0.168965 | -0.174337 | 0.580552 | 0.693636 | 0.836969 | 1.486506 | 0.068026 | 1.978e+06 |
| 0.760 | 0.228741 | 0.122526 | 0.816620 | 0.428066 | 0.522731 | 0.293889 | -0.094665 | 0.524193 | 0.700207 | 0.748625 | 1.812384 | 0.066272 | 1.994e+06 |
| 0.800 | 0.164981 | 0.231439 | 0.801571 | 0.236799 | 0.396152 | 0.405419 | -0.159352 | 0.295419 | 0.575900 | 0.512970 | 2.209701 | 0.102617 | 1.992e+06 |
| 0.840 | 0.135266 | 0.244761 | 0.780494 | 0.244789 | 0.260962 | 0.519532 | -0.016173 | 0.313634 | 0.650254 | 0.482325 | 2.694119 | 0.080036 | 2.003e+06 |
| 0.880 | 0.123576 | 0.214576 | 0.781520 | 0.109488 | 0.165904 | 0.615616 | -0.056417 | 0.140096 | 0.549716 | 0.254851 | 3.284734 | 0.111270 | 2.005e+06 |
| 0.960 | 0.129431 | 0.333750 | 0.726021 | 0.101205 | 0.076899 | 0.649122 | 0.024305 | 0.139396 | 0.611030 | 0.228133 | 4.000000 | 0.091605 | 2.016e+06 |

### Output Artifacts

| Artifact | Path |
| --- | --- |
| anchor_evidence_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_anchor_evidence_20260621_121054.csv |
| checkpoint_manifest_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_checkpoint_manifest_20260621_121054.csv |
| dynamic_trajectory_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_dynamic_trajectory_20260621_121054.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_evidence_20260621_121054.json |
| freeze_manifest_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_freeze_manifest_20260621_121054.json |
| live_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_live_insertion_rows_20260621_121054.csv |
| null_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_null_insertion_rows_20260621_121054.csv |
| out_dir | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054 |
| projection_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\projection_base.npz |
| smallN_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\base_smallN_L2_q8.npz |
| temp_curve_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_temp_curve_20260621_121054.csv |
| trajectory_summary_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_trajectory_summary_20260621_121054.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\unified_main_pipeline\20260621_121054\unified_evidence_20260621_121054.json |
| runner_log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\03_q512_smoke_insertion_decay\03_q512_smoke_insertion_decay.log |

### Freeze And Dynamic Outputs

| Field | Value |
| --- | --- |
| freeze_passed | True |
| Pi_lambda0 | 0.2646686977185224 |
| Pi_anchor_count | None |
| Pi_candidate_count | None |
| causal_tap_lags | None |
| causal_tap_weights | None |
| resolution_gate_passed | True |
| dynamic_gates | {"dynamic_diverged": true, "live_post_state_hamming_final_total": 35, "max_abs_live_minus_post_final": 0.0009455285024523308, "mean_live_minus_post_final": -9.817205461840409e-05, "null_matches_post_final": true, "null_post_state_hamming_final_total": 0} |
| live_summary | {"kicked_trajectories": 11, "lambda0": 0.2646686977185224, "max_abs_kick_float": 1.5000000000000002, "total_kicks": 29} |
| null_summary | {"kicked_trajectories": 0, "lambda0": 0.04322922062735866, "max_abs_kick_float": 0.245, "total_kicks": 0} |
| live_post_hamming | 35 |

## Test 04: q512_mid_freeze

### Raw Console Output

```text
[UNIFIED] T=0.4000 |Psi_g|=5.2611e-01 chi=1.0193e-01 pairUPS=3.841e+06
[UNIFIED] T=0.5600 |Psi_g|=4.9491e-01 chi=1.2350e-01 pairUPS=3.846e+06
[UNIFIED] T=0.6400 |Psi_g|=3.6460e-01 chi=1.5664e-01 pairUPS=3.827e+06
[UNIFIED] T=0.7200 |Psi_g|=2.7909e-01 chi=2.5287e-01 pairUPS=3.835e+06
[UNIFIED] T=0.7600 |Psi_g|=3.0380e-01 chi=2.9399e-01 pairUPS=3.854e+06
[UNIFIED] T=0.8000 |Psi_g|=2.5634e-01 chi=4.0374e-01 pairUPS=3.871e+06
[UNIFIED] T=0.8400 |Psi_g|=2.1994e-01 chi=4.2018e-01 pairUPS=3.890e+06
[UNIFIED] T=0.8800 |Psi_g|=2.1436e-01 chi=4.8930e-01 pairUPS=3.897e+06
[UNIFIED] T=0.9600 |Psi_g|=1.6539e-01 chi=8.7665e-01 pairUPS=3.897e+06
[UNIFIED] passed              = True
[UNIFIED] projection base     = analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\projection_base.npz
[UNIFIED] T_ins               = 0.800000
[UNIFIED] scout score         = 4.008557e-02
[UNIFIED] insertion decay     = 7.082361e-01
[UNIFIED] lambda0             = 1.991697e-01
[UNIFIED] K2 live kicks       = 50
[UNIFIED] live-post hamming   = 64
[UNIFIED] evidence            = analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_evidence_20260621_121148.json
```

### Console Summary

| Field | Value |
| --- | --- |
| passed | True |
| projection base | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\projection_base.npz |
| T_ins | 0.800000 |
| scout score | 0.040086 |
| insertion decay | 0.708236 |
| lambda0 | 0.199170 |
| K2 live kicks | 50 |
| live-post hamming | 64 |
| evidence | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_evidence_20260621_121148.json |
| log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\04_q512_mid_freeze.log |

### Run Configuration

| Config | Value |
| --- | --- |
| L | 8 |
| q | 512 |
| n_traj | 64 |
| repeats | 1 |
| burn_steps | 300000 |
| sample_steps | 110000 |
| sample_stride | 2000 |
| checkpoint_steps | 11000 |
| forward_steps | 30000 |
| forward_sample_stride | 1000 |
| temps | 0.40,0.56,0.64,0.72,0.76,0.80,0.84,0.88,0.96 |
| Psi_scale_pi | 0.85 |
| proposal_id | 2 |
| proposal_delta_rad | 0.1 |
| window_radius | 1 |
| init | ordered |
| seed | 1234 |
| forward_seed | 987654 |
| threads_per_block | 128 |
| tap_lags | 1,2,3 |
| tap_weights | 0.5,0.3,0.2 |
| beta_pi | 0.5 |
| rho | 0.1 |
| lambda_safety_factor | 3.0 |
| null_scale | 0.49 |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | None |

### Core Metrics

| Metric | Value |
| --- | --- |
| raw_mae_vs_author | 0.28408020 |
| insertable_mae_vs_author | 0.08046368 |
| insertable_max_abs_error | 0.20311031 |
| insertable_bias | 0.08046368 |
| insertable_rmse | 0.10695649 |
| K1_max_abs_error | 4.17257794e-16 |
| K1_passed | True |
| selector_is_edge | False |
| selector_score | 0.04008557 |
| K2_live_total_kicks | 50 |
| K2_live_kicked_trajectories | 26 |
| K2_live_max_abs_kick_float | 1.50000000 |
| K2_null_total_kicks | 0 |
| live_post_hamming | 64 |

### Causal Insertion Decay

| Field | Value |
| --- | --- |
| stage | causal_insertion_decay_v7 |
| enabled | True |
| uses_author_data | False |
| decay_formula | fit log(coherence_lag)=a-rate*lag across causal taps; rate_excess=max(0, rate(T)-min_T rate(T)); lag_slope_decay=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints); thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max), with T_guide=min(T_ins, simulated chi shoulder); thermal_envelope_factor=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints*(thermal_gain(T)-1)); D_insert(T)=min(1, lag_slope_decay*thermal_envelope_factor); auto horizon=max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps; Psi_Y still uses coherence-thinned taps for K2 freeze; finite-clock insertion_threshold_survival is recorded as a K2 readiness diagnostic, not multiplied into the observable; c_mean_m_insertable=c_mean_m_raw*D_insert(T); lag_strength and default kappa are frozen global finite-horizon calibration constants |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | 0.925 |
| calibration_note | single frozen global finite-horizon calibration; no runtime author-data lookup |
| guide_center_T | 0.8 |
| guide_center_mode | min(T_ins, chi_shoulder) |
| chi_shoulder_T | 0.88 |
| guide_iteration | 1 |
| guide_fixed_point_converged | True |
| guide_width | 0.18666666666666665 |
| guide_strength_mode | frozen_global_default |
| horizon_mode | auto |
| horizon_checkpoints | 7.427272727272728 |
| rate_floor_rule | minimum fitted lag-coherence decay rate across the simulated temperature grid |
| threshold_role | diagnostic/readiness gate for K2 insertion, not an observable multiplier |

### Scout Selector Rows

| T | score | |Psi_g| | chi | c_mean_m | disorder | decay |
| --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.00000000 | 0.52611174 | 0.10192948 | 0.91315694 | 0.08684306 | 1.00000000 |
| 0.560 | 0.00489442 | 0.49490670 | 0.12350314 | 0.86412643 | 0.13587357 | 0.97496092 |
| 0.640 | 0.00941929 | 0.36459644 | 0.15663785 | 0.84690436 | 0.15309564 | 0.90575345 |
| 0.720 | 0.02197995 | 0.27909471 | 0.25287142 | 0.80496944 | 0.19503056 | 0.82188324 |
| 0.760 | 0.03493496 | 0.30380096 | 0.29398670 | 0.79513781 | 0.20486219 | 0.77313789 |
| 0.800 | 0.04008557 | 0.25633992 | 0.40373574 | 0.76999532 | 0.23000468 | 0.70823606 |
| 0.840 | 0.02577571 | 0.21994346 | 0.42018020 | 0.73140329 | 0.26859671 | 0.56712276 |
| 0.880 | 0.02538422 | 0.21435588 | 0.48929747 | 0.72947963 | 0.27052037 | 0.50586030 |
| 0.960 | 0.00000000 | 0.16538814 | 0.87664825 | 0.66470785 | 0.33529215 | 0.23936628 |

### Temperature Curve Rows

| T | |Psi_g| | chi | raw M | insertable M | author M | raw err | insert err | decay | lag decay | thermal env | guide gain | excess rate | pair UPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.526112 | 0.101929 | 0.913157 | 0.913157 | 0.900455 | 0.012702 | 0.012702 | 1.000000 | 1.000000 | 1.000000 | 0.250000 | 0.000000 | 3.841e+06 |
| 0.560 | 0.494907 | 0.123503 | 0.864126 | 0.842490 | 0.827175 | 0.036951 | 0.015314 | 0.974961 | 0.920081 | 1.059647 | 0.304439 | 0.013593 | 3.846e+06 |
| 0.640 | 0.364596 | 0.156638 | 0.846904 | 0.767087 | 0.763384 | 0.083521 | 0.003703 | 0.905753 | 0.803535 | 1.127211 | 0.452550 | 0.035697 | 3.827e+06 |
| 0.720 | 0.279095 | 0.252871 | 0.804969 | 0.661591 | 0.649498 | 0.155472 | 0.012093 | 0.821883 | 0.747076 | 1.100134 | 0.672718 | 0.047587 | 3.835e+06 |
| 0.760 | 0.303801 | 0.293987 | 0.795138 | 0.614751 | 0.522731 | 0.272407 | 0.092020 | 0.773138 | 0.730736 | 1.058027 | 0.820194 | 0.051196 | 3.854e+06 |
| 0.800 | 0.256340 | 0.403736 | 0.769995 | 0.545338 | 0.396152 | 0.373844 | 0.149187 | 0.708236 | 0.708236 | 1.000000 | 1.000000 | 0.056300 | 3.871e+06 |
| 0.840 | 0.219943 | 0.420180 | 0.731403 | 0.414795 | 0.260962 | 0.470441 | 0.153834 | 0.567123 | 0.628011 | 0.903046 | 1.219224 | 0.075920 | 3.890e+06 |
| 0.880 | 0.214356 | 0.489297 | 0.729480 | 0.369015 | 0.165904 | 0.563575 | 0.203110 | 0.505860 | 0.632260 | 0.800082 | 1.486506 | 0.074819 | 3.897e+06 |
| 0.960 | 0.165388 | 0.876648 | 0.664708 | 0.159109 | 0.076899 | 0.587808 | 0.082209 | 0.239366 | 0.523594 | 0.457160 | 2.209701 | 0.105596 | 3.897e+06 |

### Output Artifacts

| Artifact | Path |
| --- | --- |
| anchor_evidence_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_anchor_evidence_20260621_121148.csv |
| checkpoint_manifest_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_checkpoint_manifest_20260621_121148.csv |
| dynamic_trajectory_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_dynamic_trajectory_20260621_121148.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_evidence_20260621_121148.json |
| freeze_manifest_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_freeze_manifest_20260621_121148.json |
| live_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_live_insertion_rows_20260621_121148.csv |
| null_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_null_insertion_rows_20260621_121148.csv |
| out_dir | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148 |
| projection_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\projection_base.npz |
| smallN_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\base_smallN_L2_q8.npz |
| temp_curve_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_temp_curve_20260621_121148.csv |
| trajectory_summary_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_trajectory_summary_20260621_121148.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\unified_main_pipeline\20260621_121148\unified_evidence_20260621_121148.json |
| runner_log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\04_q512_mid_freeze\04_q512_mid_freeze.log |

### Freeze And Dynamic Outputs

| Field | Value |
| --- | --- |
| freeze_passed | True |
| Pi_lambda0 | 0.19916972100560476 |
| Pi_anchor_count | None |
| Pi_candidate_count | None |
| causal_tap_lags | None |
| causal_tap_weights | None |
| resolution_gate_passed | True |
| dynamic_gates | {"dynamic_diverged": true, "live_post_state_hamming_final_total": 64, "max_abs_live_minus_post_final": 0.004235527889889457, "mean_live_minus_post_final": -4.1953670552576094e-05, "null_matches_post_final": true, "null_post_state_hamming_final_total": 0} |
| live_summary | {"kicked_trajectories": 26, "lambda0": 0.19916972100560476, "max_abs_kick_float": 1.5, "total_kicks": 50} |
| null_summary | {"kicked_trajectories": 0, "lambda0": 0.032531054430915446, "max_abs_kick_float": 0.245, "total_kicks": 0} |
| live_post_hamming | 64 |

## Test 05: q512_frozen_candidate

### Raw Console Output

```text
[UNIFIED] T=0.4000 |Psi_g|=6.7883e-01 chi=8.7903e-02 pairUPS=3.876e+06
[UNIFIED] T=0.5600 |Psi_g|=6.1873e-01 chi=1.7270e-01 pairUPS=3.890e+06
[UNIFIED] T=0.6400 |Psi_g|=5.0325e-01 chi=2.0995e-01 pairUPS=3.888e+06
[UNIFIED] T=0.7200 |Psi_g|=3.9966e-01 chi=2.2935e-01 pairUPS=3.887e+06
[UNIFIED] T=0.7600 |Psi_g|=3.8697e-01 chi=3.2568e-01 pairUPS=3.907e+06
[UNIFIED] T=0.8000 |Psi_g|=3.6762e-01 chi=3.8989e-01 pairUPS=3.931e+06
[UNIFIED] T=0.8400 |Psi_g|=3.1468e-01 chi=6.7226e-01 pairUPS=3.948e+06
[UNIFIED] T=0.8800 |Psi_g|=2.9713e-01 chi=6.3799e-01 pairUPS=3.953e+06
[UNIFIED] T=0.9600 |Psi_g|=2.5057e-01 chi=1.1619e+00 pairUPS=3.961e+06
[UNIFIED] passed              = True
[UNIFIED] projection base     = analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\projection_base.npz
[UNIFIED] T_ins               = 0.840000
[UNIFIED] scout score         = 2.999213e-02
[UNIFIED] insertion decay     = 5.268798e-01
[UNIFIED] lambda0             = 1.719091e-01
[UNIFIED] K2 live kicks       = 61
[UNIFIED] live-post hamming   = 101
[UNIFIED] evidence            = analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_evidence_20260621_121357.json
```

### Console Summary

| Field | Value |
| --- | --- |
| passed | True |
| projection base | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\projection_base.npz |
| T_ins | 0.840000 |
| scout score | 0.029992 |
| insertion decay | 0.526880 |
| lambda0 | 0.171909 |
| K2 live kicks | 61 |
| live-post hamming | 101 |
| evidence | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_evidence_20260621_121357.json |
| log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\05_q512_frozen_candidate.log |

### Run Configuration

| Config | Value |
| --- | --- |
| L | 8 |
| q | 512 |
| n_traj | 64 |
| repeats | 1 |
| burn_steps | 600000 |
| sample_steps | 220000 |
| sample_stride | 2000 |
| checkpoint_steps | 22000 |
| forward_steps | 50000 |
| forward_sample_stride | 1000 |
| temps | 0.40,0.56,0.64,0.72,0.76,0.80,0.84,0.88,0.96 |
| Psi_scale_pi | 0.85 |
| proposal_id | 2 |
| proposal_delta_rad | 0.1 |
| window_radius | 1 |
| init | ordered |
| seed | 1234 |
| forward_seed | 987654 |
| threads_per_block | 128 |
| tap_lags | 1,2,3 |
| tap_weights | 0.5,0.3,0.2 |
| beta_pi | 0.5 |
| rho | 0.1 |
| lambda_safety_factor | 3.0 |
| null_scale | 0.49 |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | None |

### Core Metrics

| Metric | Value |
| --- | --- |
| raw_mae_vs_author | 0.27265815 |
| insertable_mae_vs_author | 0.04708507 |
| insertable_max_abs_error | 0.11692843 |
| insertable_bias | 0.04089970 |
| insertable_rmse | 0.06283667 |
| K1_max_abs_error | 4.96506831e-16 |
| K1_passed | True |
| selector_is_edge | False |
| selector_score | 0.02999213 |
| K2_live_total_kicks | 61 |
| K2_live_kicked_trajectories | 29 |
| K2_live_max_abs_kick_float | 1.50000000 |
| K2_null_total_kicks | 0 |
| live_post_hamming | 101 |

### Causal Insertion Decay

| Field | Value |
| --- | --- |
| stage | causal_insertion_decay_v7 |
| enabled | True |
| uses_author_data | False |
| decay_formula | fit log(coherence_lag)=a-rate*lag across causal taps; rate_excess=max(0, rate(T)-min_T rate(T)); lag_slope_decay=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints); thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max), with T_guide=min(T_ins, simulated chi shoulder); thermal_envelope_factor=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints*(thermal_gain(T)-1)); D_insert(T)=min(1, lag_slope_decay*thermal_envelope_factor); auto horizon=max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps; Psi_Y still uses coherence-thinned taps for K2 freeze; finite-clock insertion_threshold_survival is recorded as a K2 readiness diagnostic, not multiplied into the observable; c_mean_m_insertable=c_mean_m_raw*D_insert(T); lag_strength and default kappa are frozen global finite-horizon calibration constants |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | 0.925 |
| calibration_note | single frozen global finite-horizon calibration; no runtime author-data lookup |
| guide_center_T | 0.84 |
| guide_center_mode | min(T_ins, chi_shoulder) |
| chi_shoulder_T | 0.84 |
| guide_iteration | 1 |
| guide_fixed_point_converged | True |
| guide_width | 0.18666666666666665 |
| guide_strength_mode | frozen_global_default |
| horizon_mode | auto |
| horizon_checkpoints | 6.972727272727273 |
| rate_floor_rule | minimum fitted lag-coherence decay rate across the simulated temperature grid |
| threshold_role | diagnostic/readiness gate for K2 insertion, not an observable multiplier |

### Scout Selector Rows

| T | score | |Psi_g| | chi | c_mean_m | disorder | decay |
| --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.00000000 | 0.67883445 | 0.08790266 | 0.91233328 | 0.08766672 | 1.00000000 |
| 0.560 | 0.01147629 | 0.61872880 | 0.17270371 | 0.86336358 | 0.13663642 | 0.96376494 |
| 0.640 | 0.01687408 | 0.50325153 | 0.20994993 | 0.83486828 | 0.16513172 | 0.90687757 |
| 0.720 | 0.01346335 | 0.39966380 | 0.22935118 | 0.80699204 | 0.19300796 | 0.77810431 |
| 0.760 | 0.02466170 | 0.38697411 | 0.32567994 | 0.78588382 | 0.21411618 | 0.77205915 |
| 0.800 | 0.02554312 | 0.36762038 | 0.38989478 | 0.75814686 | 0.24185314 | 0.60173329 |
| 0.840 | 0.02999213 | 0.31467573 | 0.67225898 | 0.71722319 | 0.28277681 | 0.52687976 |
| 0.880 | 0.01565996 | 0.29712502 | 0.63799170 | 0.70507397 | 0.29492603 | 0.37883255 |
| 0.960 | 0.00000000 | 0.25057452 | 1.16190503 | 0.63319767 | 0.36680233 | 0.14821619 |

### Temperature Curve Rows

| T | |Psi_g| | chi | raw M | insertable M | author M | raw err | insert err | decay | lag decay | thermal env | guide gain | excess rate | pair UPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.678834 | 0.087903 | 0.912333 | 0.912333 | 0.900455 | 0.011879 | 0.011879 | 1.000000 | 1.000000 | 1.000000 | 0.250000 | 0.000000 | 3.876e+06 |
| 0.560 | 0.618729 | 0.172704 | 0.863364 | 0.832080 | 0.827175 | 0.036188 | 0.004904 | 0.963765 | 0.862749 | 1.117086 | 0.250000 | 0.025664 | 3.890e+06 |
| 0.640 | 0.503252 | 0.209950 | 0.834868 | 0.757123 | 0.763384 | 0.071485 | -0.006260 | 0.906878 | 0.768477 | 1.180097 | 0.371179 | 0.045779 | 3.888e+06 |
| 0.720 | 0.399664 | 0.229351 | 0.806992 | 0.627924 | 0.649498 | 0.157494 | -0.021574 | 0.778104 | 0.634627 | 1.226081 | 0.551760 | 0.079047 | 3.887e+06 |
| 0.760 | 0.386974 | 0.325680 | 0.785884 | 0.606749 | 0.522731 | 0.263153 | 0.084018 | 0.772059 | 0.680757 | 1.134119 | 0.672718 | 0.066849 | 3.907e+06 |
| 0.800 | 0.367620 | 0.389895 | 0.758147 | 0.456202 | 0.396152 | 0.361995 | 0.060051 | 0.601733 | 0.538325 | 1.117789 | 0.820194 | 0.107656 | 3.931e+06 |
| 0.840 | 0.314676 | 0.672259 | 0.717223 | 0.377890 | 0.260962 | 0.456261 | 0.116928 | 0.526880 | 0.526880 | 1.000000 | 1.000000 | 0.111392 | 3.948e+06 |
| 0.880 | 0.297125 | 0.637992 | 0.705074 | 0.267105 | 0.165904 | 0.539169 | 0.101200 | 0.378833 | 0.451071 | 0.839851 | 1.219224 | 0.138397 | 3.953e+06 |
| 0.960 | 0.250575 | 1.161905 | 0.633198 | 0.093850 | 0.076899 | 0.556298 | 0.016951 | 0.148216 | 0.348766 | 0.424974 | 1.812384 | 0.183113 | 3.961e+06 |

### Output Artifacts

| Artifact | Path |
| --- | --- |
| anchor_evidence_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_anchor_evidence_20260621_121357.csv |
| checkpoint_manifest_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_checkpoint_manifest_20260621_121357.csv |
| dynamic_trajectory_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_dynamic_trajectory_20260621_121357.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_evidence_20260621_121357.json |
| freeze_manifest_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_freeze_manifest_20260621_121357.json |
| live_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_live_insertion_rows_20260621_121357.csv |
| null_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_null_insertion_rows_20260621_121357.csv |
| out_dir | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357 |
| projection_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\projection_base.npz |
| smallN_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\base_smallN_L2_q8.npz |
| temp_curve_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_temp_curve_20260621_121357.csv |
| trajectory_summary_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_trajectory_summary_20260621_121357.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\unified_main_pipeline\20260621_121357\unified_evidence_20260621_121357.json |
| runner_log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\05_q512_frozen_candidate\05_q512_frozen_candidate.log |

### Freeze And Dynamic Outputs

| Field | Value |
| --- | --- |
| freeze_passed | True |
| Pi_lambda0 | 0.17190910307651505 |
| Pi_anchor_count | None |
| Pi_candidate_count | None |
| causal_tap_lags | None |
| causal_tap_weights | None |
| resolution_gate_passed | True |
| dynamic_gates | {"dynamic_diverged": true, "live_post_state_hamming_final_total": 101, "max_abs_live_minus_post_final": 0.002567028597316612, "mean_live_minus_post_final": 3.7529883961547744e-05, "null_matches_post_final": true, "null_post_state_hamming_final_total": 0} |
| live_summary | {"kicked_trajectories": 29, "lambda0": 0.17190910307651505, "max_abs_kick_float": 1.5, "total_kicks": 61} |
| null_summary | {"kicked_trajectories": 0, "lambda0": 0.02807848683583079, "max_abs_kick_float": 0.245, "total_kicks": 0} |
| live_post_hamming | 101 |

## Test 06: high_temp_bracket

### Raw Console Output

```text
[UNIFIED] T=0.7200 |Psi_g|=4.3703e-01 chi=1.9826e-01 pairUPS=3.918e+06
[UNIFIED] T=0.8400 |Psi_g|=3.3644e-01 chi=7.3591e-01 pairUPS=3.940e+06
[UNIFIED] T=0.9600 |Psi_g|=2.5790e-01 chi=1.0655e+00 pairUPS=3.965e+06
[UNIFIED] T=1.0800 |Psi_g|=2.6675e-01 chi=1.5570e+00 pairUPS=3.962e+06
[UNIFIED] T=1.2000 |Psi_g|=2.5255e-01 chi=1.2607e+00 pairUPS=3.847e+06
[UNIFIED] T=1.3200 |Psi_g|=2.4425e-01 chi=9.9594e-01 pairUPS=3.883e+06
[UNIFIED] T=1.4400 |Psi_g|=2.5011e-01 chi=8.2400e-01 pairUPS=3.879e+06
[UNIFIED] passed              = True
[UNIFIED] projection base     = analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\projection_base.npz
[UNIFIED] T_ins               = 0.840000
[UNIFIED] scout score         = 2.072388e-02
[UNIFIED] insertion decay     = 7.360650e-01
[UNIFIED] lambda0             = 2.164881e-01
[UNIFIED] K2 live kicks       = 93
[UNIFIED] live-post hamming   = 146
[UNIFIED] evidence            = analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_evidence_20260621_121806.json
```

### Console Summary

| Field | Value |
| --- | --- |
| passed | True |
| projection base | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\projection_base.npz |
| T_ins | 0.840000 |
| scout score | 0.020724 |
| insertion decay | 0.736065 |
| lambda0 | 0.216488 |
| K2 live kicks | 93 |
| live-post hamming | 146 |
| evidence | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_evidence_20260621_121806.json |
| log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\06_high_temp_bracket.log |

### Run Configuration

| Config | Value |
| --- | --- |
| L | 8 |
| q | 512 |
| n_traj | 64 |
| repeats | 1 |
| burn_steps | 600000 |
| sample_steps | 220000 |
| sample_stride | 2000 |
| checkpoint_steps | 22000 |
| forward_steps | 50000 |
| forward_sample_stride | 1000 |
| temps | 0.72,0.84,0.96,1.08,1.20,1.32,1.44 |
| Psi_scale_pi | 0.85 |
| proposal_id | 2 |
| proposal_delta_rad | 0.1 |
| window_radius | 1 |
| init | ordered |
| seed | 1234 |
| forward_seed | 987654 |
| threads_per_block | 128 |
| tap_lags | 1,2,3 |
| tap_weights | 0.5,0.3,0.2 |
| beta_pi | 0.5 |
| rho | 0.1 |
| lambda_safety_factor | 3.0 |
| null_scale | 0.49 |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | None |

### Core Metrics

| Metric | Value |
| --- | --- |
| raw_mae_vs_author | 0.37970020 |
| insertable_mae_vs_author | 0.09457645 |
| insertable_max_abs_error | 0.27853007 |
| insertable_bias | 0.07526947 |
| insertable_rmse | 0.13543326 |
| K1_max_abs_error | 4.16333634e-16 |
| K1_passed | True |
| selector_is_edge | False |
| selector_score | 0.02072388 |
| K2_live_total_kicks | 93 |
| K2_live_kicked_trajectories | 40 |
| K2_live_max_abs_kick_float | 1.50000000 |
| K2_null_total_kicks | 0 |
| live_post_hamming | 146 |

### Causal Insertion Decay

| Field | Value |
| --- | --- |
| stage | causal_insertion_decay_v7 |
| enabled | True |
| uses_author_data | False |
| decay_formula | fit log(coherence_lag)=a-rate*lag across causal taps; rate_excess=max(0, rate(T)-min_T rate(T)); lag_slope_decay=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints); thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max), with T_guide=min(T_ins, simulated chi shoulder); thermal_envelope_factor=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints*(thermal_gain(T)-1)); D_insert(T)=min(1, lag_slope_decay*thermal_envelope_factor); auto horizon=max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps; Psi_Y still uses coherence-thinned taps for K2 freeze; finite-clock insertion_threshold_survival is recorded as a K2 readiness diagnostic, not multiplied into the observable; c_mean_m_insertable=c_mean_m_raw*D_insert(T); lag_strength and default kappa are frozen global finite-horizon calibration constants |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | 0.925 |
| calibration_note | single frozen global finite-horizon calibration; no runtime author-data lookup |
| guide_center_T | 0.84 |
| guide_center_mode | min(T_ins, chi_shoulder) |
| chi_shoulder_T | 0.96 |
| guide_iteration | 2 |
| guide_fixed_point_converged | True |
| guide_width | 0.24 |
| guide_strength_mode | frozen_global_default |
| horizon_mode | auto |
| horizon_checkpoints | 6.972727272727273 |
| rate_floor_rule | minimum fitted lag-coherence decay rate across the simulated temperature grid |
| threshold_role | diagnostic/readiness gate for K2 insertion, not an observable multiplier |

### Scout Selector Rows

| T | score | |Psi_g| | chi | c_mean_m | disorder | decay |
| --- | --- | --- | --- | --- | --- | --- |
| 0.720 | 0.00000000 | 0.43703283 | 0.19825716 | 0.81013974 | 0.18986026 | 1.00000000 |
| 0.840 | 0.02072388 | 0.33643766 | 0.73591272 | 0.73294070 | 0.26705930 | 0.73606503 |
| 0.960 | 0.00610088 | 0.25790101 | 1.06546920 | 0.62077591 | 0.37922409 | 0.36975864 |
| 1.080 | 0.00523990 | 0.26675145 | 1.55698932 | 0.54035413 | 0.45964587 | 0.08632338 |
| 1.200 | 0.00108838 | 0.25254608 | 1.26067539 | 0.44236643 | 0.55763357 | 0.04560903 |
| 1.320 | 0.00000000 | 0.24424553 | 0.99593962 | 0.34697959 | 0.65302041 | 0.01660170 |
| 1.440 | 9.99578063e-05 | 0.25010653 | 0.82400497 | 0.29128881 | 0.70871119 | 0.00713934 |

### Temperature Curve Rows

| T | |Psi_g| | chi | raw M | insertable M | author M | raw err | insert err | decay | lag decay | thermal env | guide gain | excess rate | pair UPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.720 | 0.437033 | 0.198257 | 0.810140 | 0.810140 | 0.649498 | 0.160642 | 0.160642 | 1.000000 | 1.000000 | 1.000000 | 0.629707 | 0.000000 | 3.918e+06 |
| 0.840 | 0.336438 | 0.735913 | 0.732941 | 0.539492 | 0.260962 | 0.471979 | 0.278530 | 0.736065 | 0.736065 | 1.000000 | 1.000000 | 0.053270 | 3.940e+06 |
| 0.960 | 0.257901 | 1.065469 | 0.620776 | 0.229537 | 0.076899 | 0.543876 | 0.152638 | 0.369759 | 0.534460 | 0.691836 | 1.588039 | 0.108909 | 3.965e+06 |
| 1.080 | 0.266751 | 1.556989 | 0.540354 | 0.046645 | 0.043994 | 0.496360 | 0.002651 | 0.086323 | 0.378566 | 0.228027 | 2.521868 | 0.168860 | 3.962e+06 |
| 1.200 | 0.252546 | 1.260675 | 0.442366 | 0.020176 | 0.031863 | 0.410503 | -0.011688 | 0.045609 | 0.462128 | 0.098693 | 4.000000 | 0.134187 | 3.847e+06 |
| 1.320 | 0.244246 | 0.995940 | 0.346980 | 0.005760 | 0.031863 | 0.315116 | -0.026103 | 0.016602 | 0.358953 | 0.046250 | 4.000000 | 0.178107 | 3.883e+06 |
| 1.440 | 0.250107 | 0.824005 | 0.291289 | 0.002080 | 0.031863 | 0.259425 | -0.029784 | 0.007139 | 0.290680 | 0.024561 | 4.000000 | 0.214782 | 3.879e+06 |

### Output Artifacts

| Artifact | Path |
| --- | --- |
| anchor_evidence_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_anchor_evidence_20260621_121806.csv |
| checkpoint_manifest_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_checkpoint_manifest_20260621_121806.csv |
| dynamic_trajectory_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_dynamic_trajectory_20260621_121806.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_evidence_20260621_121806.json |
| freeze_manifest_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_freeze_manifest_20260621_121806.json |
| live_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_live_insertion_rows_20260621_121806.csv |
| null_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_null_insertion_rows_20260621_121806.csv |
| out_dir | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806 |
| projection_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\projection_base.npz |
| smallN_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\base_smallN_L2_q8.npz |
| temp_curve_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_temp_curve_20260621_121806.csv |
| trajectory_summary_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_trajectory_summary_20260621_121806.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\unified_main_pipeline\20260621_121806\unified_evidence_20260621_121806.json |
| runner_log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\06_high_temp_bracket\06_high_temp_bracket.log |

### Freeze And Dynamic Outputs

| Field | Value |
| --- | --- |
| freeze_passed | True |
| Pi_lambda0 | 0.216488122830463 |
| Pi_anchor_count | None |
| Pi_candidate_count | None |
| causal_tap_lags | None |
| causal_tap_weights | None |
| resolution_gate_passed | True |
| dynamic_gates | {"dynamic_diverged": true, "live_post_state_hamming_final_total": 146, "max_abs_live_minus_post_final": 0.0066514344816487325, "mean_live_minus_post_final": 5.27398585720637e-05, "null_matches_post_final": true, "null_post_state_hamming_final_total": 0} |
| live_summary | {"kicked_trajectories": 40, "lambda0": 0.216488122830463, "max_abs_kick_float": 1.5, "total_kicks": 93} |
| null_summary | {"kicked_trajectories": 0, "lambda0": 0.035359726728975624, "max_abs_kick_float": 0.245, "total_kicks": 0} |
| live_post_hamming | 146 |

## Test 07: q1024_continuum_check

### Raw Console Output

```text
[UNIFIED] T=0.4000 |Psi_g|=5.2574e-01 chi=1.0010e-01 pairUPS=3.885e+06
[UNIFIED] T=0.5600 |Psi_g|=4.7080e-01 chi=1.2248e-01 pairUPS=3.900e+06
[UNIFIED] T=0.6400 |Psi_g|=3.5086e-01 chi=1.5271e-01 pairUPS=3.926e+06
[UNIFIED] T=0.7200 |Psi_g|=2.7298e-01 chi=2.7663e-01 pairUPS=3.914e+06
[UNIFIED] T=0.7600 |Psi_g|=2.7677e-01 chi=2.7510e-01 pairUPS=3.880e+06
[UNIFIED] T=0.8000 |Psi_g|=1.9990e-01 chi=3.7922e-01 pairUPS=4.004e+06
[UNIFIED] T=0.8400 |Psi_g|=2.0395e-01 chi=3.8466e-01 pairUPS=4.010e+06
[UNIFIED] T=0.8800 |Psi_g|=1.8270e-01 chi=5.2786e-01 pairUPS=4.075e+06
[UNIFIED] T=0.9600 |Psi_g|=1.8354e-01 chi=8.9877e-01 pairUPS=4.089e+06
[UNIFIED] passed              = True
[UNIFIED] projection base     = analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\projection_base.npz
[UNIFIED] T_ins               = 0.760000
[UNIFIED] scout score         = 2.004729e-02
[UNIFIED] insertion decay     = 6.967719e-01
[UNIFIED] lambda0             = 8.872485e-02
[UNIFIED] K2 live kicks       = 55
[UNIFIED] live-post hamming   = 68
[UNIFIED] evidence            = analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_evidence_20260621_122122.json
```

### Console Summary

| Field | Value |
| --- | --- |
| passed | True |
| projection base | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\projection_base.npz |
| T_ins | 0.760000 |
| scout score | 0.020047 |
| insertion decay | 0.696772 |
| lambda0 | 0.088725 |
| K2 live kicks | 55 |
| live-post hamming | 68 |
| evidence | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_evidence_20260621_122122.json |
| log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\07_q1024_continuum_check.log |

### Run Configuration

| Config | Value |
| --- | --- |
| L | 8 |
| q | 1024 |
| n_traj | 64 |
| repeats | 1 |
| burn_steps | 300000 |
| sample_steps | 110000 |
| sample_stride | 2000 |
| checkpoint_steps | 11000 |
| forward_steps | 30000 |
| forward_sample_stride | 1000 |
| temps | 0.40,0.56,0.64,0.72,0.76,0.80,0.84,0.88,0.96 |
| Psi_scale_pi | 0.85 |
| proposal_id | 2 |
| proposal_delta_rad | 0.1 |
| window_radius | 1 |
| init | ordered |
| seed | 1234 |
| forward_seed | 987654 |
| threads_per_block | 128 |
| tap_lags | 1,2,3 |
| tap_weights | 0.5,0.3,0.2 |
| beta_pi | 0.5 |
| rho | 0.1 |
| lambda_safety_factor | 3.0 |
| null_scale | 0.49 |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | None |

### Core Metrics

| Metric | Value |
| --- | --- |
| raw_mae_vs_author | 0.28478341 |
| insertable_mae_vs_author | 0.04027561 |
| insertable_max_abs_error | 0.08755722 |
| insertable_bias | 0.02329216 |
| insertable_rmse | 0.04845283 |
| K1_max_abs_error | 4.99600361e-16 |
| K1_passed | True |
| selector_is_edge | False |
| selector_score | 0.02004729 |
| K2_live_total_kicks | 55 |
| K2_live_kicked_trajectories | 25 |
| K2_live_max_abs_kick_float | 1.50000000 |
| K2_null_total_kicks | 0 |
| live_post_hamming | 68 |

### Causal Insertion Decay

| Field | Value |
| --- | --- |
| stage | causal_insertion_decay_v7 |
| enabled | True |
| uses_author_data | False |
| decay_formula | fit log(coherence_lag)=a-rate*lag across causal taps; rate_excess=max(0, rate(T)-min_T rate(T)); lag_slope_decay=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints); thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max), with T_guide=min(T_ins, simulated chi shoulder); thermal_envelope_factor=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints*(thermal_gain(T)-1)); D_insert(T)=min(1, lag_slope_decay*thermal_envelope_factor); auto horizon=max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps; Psi_Y still uses coherence-thinned taps for K2 freeze; finite-clock insertion_threshold_survival is recorded as a K2 readiness diagnostic, not multiplied into the observable; c_mean_m_insertable=c_mean_m_raw*D_insert(T); lag_strength and default kappa are frozen global finite-horizon calibration constants |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | 0.925 |
| calibration_note | single frozen global finite-horizon calibration; no runtime author-data lookup |
| guide_center_T | 0.76 |
| guide_center_mode | min(T_ins, chi_shoulder) |
| chi_shoulder_T | 0.88 |
| guide_iteration | 1 |
| guide_fixed_point_converged | True |
| guide_width | 0.18666666666666665 |
| guide_strength_mode | frozen_global_default |
| horizon_mode | auto |
| horizon_checkpoints | 7.427272727272728 |
| rate_floor_rule | minimum fitted lag-coherence decay rate across the simulated temperature grid |
| threshold_role | diagnostic/readiness gate for K2 insertion, not an observable multiplier |

### Scout Selector Rows

| T | score | |Psi_g| | chi | c_mean_m | disorder | decay |
| --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.00000000 | 0.52574027 | 0.10009698 | 0.91334526 | 0.08665474 | 1.00000000 |
| 0.560 | 0.00461416 | 0.47080191 | 0.12247637 | 0.86295902 | 0.13704098 | 0.95715112 |
| 0.640 | 0.00763170 | 0.35085860 | 0.15271475 | 0.84719449 | 0.15280551 | 0.87867506 |
| 0.720 | 0.01929621 | 0.27298382 | 0.27663188 | 0.80292051 | 0.19707949 | 0.73885288 |
| 0.760 | 0.02004729 | 0.27676762 | 0.27510041 | 0.79556536 | 0.20443464 | 0.69677186 |
| 0.800 | 0.00559277 | 0.19989888 | 0.37922284 | 0.77059545 | 0.22940455 | 0.54996967 |
| 0.840 | 0.00695572 | 0.20395159 | 0.38465883 | 0.73555069 | 0.26444931 | 0.43595790 |
| 0.880 | 0.00000000 | 0.18269976 | 0.52786227 | 0.73070360 | 0.26929640 | 0.34687349 |
| 0.960 | 5.24323632e-04 | 0.18353596 | 0.89876656 | 0.66737564 | 0.33262436 | 0.21509613 |

### Temperature Curve Rows

| T | |Psi_g| | chi | raw M | insertable M | author M | raw err | insert err | decay | lag decay | thermal env | guide gain | excess rate | pair UPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.525740 | 0.100097 | 0.913345 | 0.913345 | 0.900455 | 0.012891 | 0.012891 | 1.000000 | 1.000000 | 1.000000 | 0.250000 | 0.000000 | 3.885e+06 |
| 0.560 | 0.470802 | 0.122476 | 0.862959 | 0.825982 | 0.827175 | 0.035784 | -0.001193 | 0.957151 | 0.888708 | 1.077014 | 0.371179 | 0.019255 | 3.900e+06 |
| 0.640 | 0.350859 | 0.152715 | 0.847194 | 0.744409 | 0.763384 | 0.083811 | -0.018975 | 0.878675 | 0.791034 | 1.110793 | 0.551760 | 0.038256 | 3.926e+06 |
| 0.720 | 0.272984 | 0.276632 | 0.802921 | 0.593240 | 0.649498 | 0.153423 | -0.056258 | 0.738853 | 0.691421 | 1.068600 | 0.820194 | 0.060221 | 3.914e+06 |
| 0.760 | 0.276768 | 0.275100 | 0.795565 | 0.554328 | 0.522731 | 0.272835 | 0.031597 | 0.696772 | 0.696772 | 1.000000 | 1.000000 | 0.058963 | 3.880e+06 |
| 0.800 | 0.199899 | 0.379223 | 0.770595 | 0.423804 | 0.396152 | 0.374444 | 0.027653 | 0.549970 | 0.612389 | 0.898072 | 1.219224 | 0.080031 | 4.004e+06 |
| 0.840 | 0.203952 | 0.384659 | 0.735551 | 0.320669 | 0.260962 | 0.474589 | 0.059707 | 0.435958 | 0.572068 | 0.762073 | 1.486506 | 0.091146 | 4.010e+06 |
| 0.880 | 0.182700 | 0.527862 | 0.730704 | 0.253462 | 0.165904 | 0.564799 | 0.087557 | 0.346873 | 0.557552 | 0.622137 | 1.812384 | 0.095341 | 4.075e+06 |
| 0.960 | 0.183536 | 0.898767 | 0.667376 | 0.143550 | 0.076899 | 0.590476 | 0.066650 | 0.215096 | 0.565311 | 0.380492 | 2.694119 | 0.093085 | 4.089e+06 |

### Output Artifacts

| Artifact | Path |
| --- | --- |
| anchor_evidence_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_anchor_evidence_20260621_122122.csv |
| checkpoint_manifest_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_checkpoint_manifest_20260621_122122.csv |
| dynamic_trajectory_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_dynamic_trajectory_20260621_122122.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_evidence_20260621_122122.json |
| freeze_manifest_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_freeze_manifest_20260621_122122.json |
| live_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_live_insertion_rows_20260621_122122.csv |
| null_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_null_insertion_rows_20260621_122122.csv |
| out_dir | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122 |
| projection_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\projection_base.npz |
| smallN_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\base_smallN_L2_q8.npz |
| temp_curve_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_temp_curve_20260621_122122.csv |
| trajectory_summary_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_trajectory_summary_20260621_122122.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\unified_main_pipeline\20260621_122122\unified_evidence_20260621_122122.json |
| runner_log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\07_q1024_continuum_check\07_q1024_continuum_check.log |

### Freeze And Dynamic Outputs

| Field | Value |
| --- | --- |
| freeze_passed | True |
| Pi_lambda0 | 0.08872485427847561 |
| Pi_anchor_count | None |
| Pi_candidate_count | None |
| causal_tap_lags | None |
| causal_tap_weights | None |
| resolution_gate_passed | True |
| dynamic_gates | {"dynamic_diverged": true, "live_post_state_hamming_final_total": 68, "max_abs_live_minus_post_final": 0.002833502204213856, "mean_live_minus_post_final": -3.6573299364069806e-06, "null_matches_post_final": true, "null_post_state_hamming_final_total": 0} |
| live_summary | {"kicked_trajectories": 25, "lambda0": 0.08872485427847561, "max_abs_kick_float": 1.5, "total_kicks": 55} |
| null_summary | {"kicked_trajectories": 0, "lambda0": 0.014491726198817683, "max_abs_kick_float": 0.245, "total_kicks": 0} |
| live_post_hamming | 68 |

## Test 08: L16_medium

### Raw Console Output

```text
[UNIFIED] T=0.4000 |Psi_g|=3.8660e-01 chi=1.4118e-01 pairUPS=3.756e+06
[UNIFIED] T=0.5600 |Psi_g|=3.0578e-01 chi=3.3100e-01 pairUPS=3.725e+06
[UNIFIED] T=0.6400 |Psi_g|=2.4127e-01 chi=5.0050e-01 pairUPS=3.738e+06
[UNIFIED] T=0.7200 |Psi_g|=1.9935e-01 chi=9.3345e-01 pairUPS=3.747e+06
[UNIFIED] T=0.7600 |Psi_g|=1.8674e-01 chi=9.8868e-01 pairUPS=3.764e+06
[UNIFIED] T=0.8000 |Psi_g|=1.5876e-01 chi=1.0608e+00 pairUPS=3.805e+06
[UNIFIED] T=0.8400 |Psi_g|=1.4818e-01 chi=8.3989e-01 pairUPS=3.904e+06
[UNIFIED] T=0.8800 |Psi_g|=1.3989e-01 chi=1.2461e+00 pairUPS=3.908e+06
[UNIFIED] T=0.9600 |Psi_g|=1.2943e-01 chi=1.5360e+00 pairUPS=3.922e+06
[UNIFIED] passed              = True
[UNIFIED] projection base     = analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\projection_base.npz
[UNIFIED] T_ins               = 0.720000
[UNIFIED] scout score         = 4.767214e-02
[UNIFIED] insertion decay     = 6.765028e-01
[UNIFIED] lambda0             = 1.000849e+00
[UNIFIED] K2 live kicks       = 93
[UNIFIED] live-post hamming   = 106
[UNIFIED] evidence            = analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_evidence_20260621_122328.json
```

### Console Summary

| Field | Value |
| --- | --- |
| passed | True |
| projection base | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\projection_base.npz |
| T_ins | 0.720000 |
| scout score | 0.047672 |
| insertion decay | 0.676503 |
| lambda0 | 1.000849 |
| K2 live kicks | 93 |
| live-post hamming | 106 |
| evidence | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_evidence_20260621_122328.json |
| log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\08_L16_medium.log |

### Run Configuration

| Config | Value |
| --- | --- |
| L | 16 |
| q | 512 |
| n_traj | 64 |
| repeats | 1 |
| burn_steps | 2000000 |
| sample_steps | 800000 |
| sample_stride | 5000 |
| checkpoint_steps | 80000 |
| forward_steps | 100000 |
| forward_sample_stride | 2000 |
| temps | 0.40,0.56,0.64,0.72,0.76,0.80,0.84,0.88,0.96 |
| Psi_scale_pi | 0.85 |
| proposal_id | 2 |
| proposal_delta_rad | 0.1 |
| window_radius | 1 |
| init | ordered |
| seed | 1234 |
| forward_seed | 987654 |
| threads_per_block | 128 |
| tap_lags | 1,2,3 |
| tap_weights | 0.5,0.3,0.2 |
| beta_pi | 0.5 |
| rho | 0.1 |
| lambda_safety_factor | 3.0 |
| null_scale | 0.49 |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | None |

### Core Metrics

| Metric | Value |
| --- | --- |
| raw_mae_vs_author | 0.23897407 |
| insertable_mae_vs_author | 0.04746641 |
| insertable_max_abs_error | 0.13301515 |
| insertable_bias | -0.04595823 |
| insertable_rmse | 0.06408952 |
| K1_max_abs_error | 5.11787527e-16 |
| K1_passed | True |
| selector_is_edge | False |
| selector_score | 0.04767214 |
| K2_live_total_kicks | 93 |
| K2_live_kicked_trajectories | 25 |
| K2_live_max_abs_kick_float | 1.50000000 |
| K2_null_total_kicks | 0 |
| live_post_hamming | 106 |

### Causal Insertion Decay

| Field | Value |
| --- | --- |
| stage | causal_insertion_decay_v7 |
| enabled | True |
| uses_author_data | False |
| decay_formula | fit log(coherence_lag)=a-rate*lag across causal taps; rate_excess=max(0, rate(T)-min_T rate(T)); lag_slope_decay=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints); thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max), with T_guide=min(T_ins, simulated chi shoulder); thermal_envelope_factor=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints*(thermal_gain(T)-1)); D_insert(T)=min(1, lag_slope_decay*thermal_envelope_factor); auto horizon=max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps; Psi_Y still uses coherence-thinned taps for K2 freeze; finite-clock insertion_threshold_survival is recorded as a K2 readiness diagnostic, not multiplied into the observable; c_mean_m_insertable=c_mean_m_raw*D_insert(T); lag_strength and default kappa are frozen global finite-horizon calibration constants |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | 0.925 |
| calibration_note | single frozen global finite-horizon calibration; no runtime author-data lookup |
| guide_center_T | 0.72 |
| guide_center_mode | min(T_ins, chi_shoulder) |
| chi_shoulder_T | 0.72 |
| guide_iteration | 2 |
| guide_fixed_point_converged | True |
| guide_width | 0.18666666666666665 |
| guide_strength_mode | frozen_global_default |
| horizon_mode | auto |
| horizon_checkpoints | 5.95 |
| rate_floor_rule | minimum fitted lag-coherence decay rate across the simulated temperature grid |
| threshold_role | diagnostic/readiness gate for K2 insertion, not an observable multiplier |

### Scout Selector Rows

| T | score | |Psi_g| | chi | c_mean_m | disorder | decay |
| --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.00000000 | 0.38660238 | 0.14117716 | 0.90413977 | 0.09586023 | 1.00000000 |
| 0.560 | 0.01669377 | 0.30578180 | 0.33099735 | 0.84526206 | 0.15473794 | 0.93664430 |
| 0.640 | 0.03090265 | 0.24126576 | 0.50050453 | 0.80452610 | 0.19547390 | 0.85368952 |
| 0.720 | 0.04767214 | 0.19934672 | 0.93345465 | 0.76345987 | 0.23654013 | 0.67650277 |
| 0.760 | 0.04249352 | 0.18674285 | 0.98867716 | 0.73919132 | 0.26080868 | 0.58651457 |
| 0.800 | 0.02209917 | 0.15875700 | 1.06075381 | 0.71109888 | 0.28890112 | 0.46942983 |
| 0.840 | 0.00912431 | 0.14818185 | 0.83989061 | 0.68942543 | 0.31057457 | 0.35867368 |
| 0.880 | 0.00598990 | 0.13989238 | 1.24609808 | 0.66096384 | 0.33903616 | 0.23564074 |
| 0.960 | 0.00000000 | 0.12943050 | 1.53600914 | 0.59585868 | 0.40414132 | 0.13426179 |

### Temperature Curve Rows

| T | |Psi_g| | chi | raw M | insertable M | author M | raw err | insert err | decay | lag decay | thermal env | guide gain | excess rate | pair UPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.386602 | 0.141177 | 0.904140 | 0.904140 | 0.900455 | 0.003685 | 0.003685 | 1.000000 | 1.000000 | 1.000000 | 0.250000 | 0.000000 | 3.756e+06 |
| 0.560 | 0.305782 | 0.330997 | 0.845262 | 0.791710 | 0.827175 | 0.018087 | -0.035465 | 0.936644 | 0.865344 | 1.082396 | 0.452550 | 0.029463 | 3.725e+06 |
| 0.640 | 0.241266 | 0.500505 | 0.804526 | 0.686815 | 0.763384 | 0.041143 | -0.076568 | 0.853690 | 0.790455 | 1.079998 | 0.672718 | 0.047904 | 3.738e+06 |
| 0.720 | 0.199347 | 0.933455 | 0.763460 | 0.516483 | 0.649498 | 0.113962 | -0.133015 | 0.676503 | 0.676503 | 1.000000 | 1.000000 | 0.079617 | 3.747e+06 |
| 0.760 | 0.186743 | 0.988677 | 0.739191 | 0.433546 | 0.522731 | 0.216461 | -0.089184 | 0.586515 | 0.645570 | 0.908521 | 1.219224 | 0.089151 | 3.764e+06 |
| 0.800 | 0.158757 | 1.060754 | 0.711099 | 0.333811 | 0.396152 | 0.314947 | -0.062341 | 0.469430 | 0.601256 | 0.780748 | 1.486506 | 0.103638 | 3.805e+06 |
| 0.840 | 0.148182 | 0.839891 | 0.689425 | 0.247279 | 0.260962 | 0.428463 | -0.013683 | 0.358674 | 0.567938 | 0.631536 | 1.812384 | 0.115252 | 3.904e+06 |
| 0.880 | 0.139892 | 1.246098 | 0.660964 | 0.155750 | 0.165904 | 0.495059 | -0.010154 | 0.235641 | 0.519891 | 0.453251 | 2.209701 | 0.133259 | 3.908e+06 |
| 0.960 | 0.129431 | 1.536009 | 0.595859 | 0.080001 | 0.076899 | 0.518959 | 0.003102 | 0.134262 | 0.542644 | 0.247421 | 3.284734 | 0.124533 | 3.922e+06 |

### Output Artifacts

| Artifact | Path |
| --- | --- |
| anchor_evidence_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_anchor_evidence_20260621_122328.csv |
| checkpoint_manifest_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_checkpoint_manifest_20260621_122328.csv |
| dynamic_trajectory_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_dynamic_trajectory_20260621_122328.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_evidence_20260621_122328.json |
| freeze_manifest_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_freeze_manifest_20260621_122328.json |
| live_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_live_insertion_rows_20260621_122328.csv |
| null_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_null_insertion_rows_20260621_122328.csv |
| out_dir | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328 |
| projection_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\projection_base.npz |
| smallN_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\base_smallN_L2_q8.npz |
| temp_curve_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_temp_curve_20260621_122328.csv |
| trajectory_summary_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_trajectory_summary_20260621_122328.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\unified_main_pipeline\20260621_122328\unified_evidence_20260621_122328.json |
| runner_log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\08_L16_medium\08_L16_medium.log |

### Freeze And Dynamic Outputs

| Field | Value |
| --- | --- |
| freeze_passed | True |
| Pi_lambda0 | 1.0008493282391386 |
| Pi_anchor_count | None |
| Pi_candidate_count | None |
| causal_tap_lags | None |
| causal_tap_weights | None |
| resolution_gate_passed | True |
| dynamic_gates | {"dynamic_diverged": true, "live_post_state_hamming_final_total": 106, "max_abs_live_minus_post_final": 0.0005236456635534514, "mean_live_minus_post_final": -1.010234594429893e-05, "null_matches_post_final": true, "null_post_state_hamming_final_total": 0} |
| live_summary | {"kicked_trajectories": 25, "lambda0": 1.0008493282391386, "max_abs_kick_float": 1.4999999999999998, "total_kicks": 93} |
| null_summary | {"kicked_trajectories": 0, "lambda0": 0.163472056945726, "max_abs_kick_float": 0.24500000000000002, "total_kicks": 0} |
| live_post_hamming | 106 |

## Test 09: L32_deep

### Raw Console Output

```text
[UNIFIED] T=0.4000 |Psi_g|=3.1439e-01 chi=1.6137e-01 pairUPS=4.583e+06
[UNIFIED] T=0.5600 |Psi_g|=2.6413e-01 chi=7.4497e-01 pairUPS=4.625e+06
[UNIFIED] T=0.6400 |Psi_g|=2.0249e-01 chi=1.0082e+00 pairUPS=4.686e+06
[UNIFIED] T=0.7200 |Psi_g|=1.6741e-01 chi=1.0279e+00 pairUPS=4.712e+06
[UNIFIED] T=0.7600 |Psi_g|=1.5702e-01 chi=1.6473e+00 pairUPS=4.732e+06
[UNIFIED] T=0.8000 |Psi_g|=1.3439e-01 chi=1.9507e+00 pairUPS=4.757e+06
[UNIFIED] T=0.8400 |Psi_g|=1.1585e-01 chi=2.2545e+00 pairUPS=4.775e+06
[UNIFIED] T=0.8800 |Psi_g|=1.0777e-01 chi=2.7799e+00 pairUPS=4.784e+06
[UNIFIED] T=0.9600 |Psi_g|=1.0124e-01 chi=4.2837e+00 pairUPS=4.802e+06
[UNIFIED] passed              = True
[UNIFIED] projection base     = analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\projection_base.npz
[UNIFIED] T_ins               = 0.760000
[UNIFIED] scout score         = 2.882443e-02
[UNIFIED] insertion decay     = 6.410935e-01
[UNIFIED] lambda0             = 6.277782e+00
[UNIFIED] K2 live kicks       = 449
[UNIFIED] live-post hamming   = 495
[UNIFIED] evidence            = analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_evidence_20260621_123747.json
```

### Console Summary

| Field | Value |
| --- | --- |
| passed | True |
| projection base | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\projection_base.npz |
| T_ins | 0.760000 |
| scout score | 0.028824 |
| insertion decay | 0.641093 |
| lambda0 | 6.277782 |
| K2 live kicks | 449 |
| live-post hamming | 495 |
| evidence | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_evidence_20260621_123747.json |
| log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\09_L32_deep.log |

### Run Configuration

| Config | Value |
| --- | --- |
| L | 32 |
| q | 512 |
| n_traj | 100 |
| repeats | 1 |
| burn_steps | 20000000 |
| sample_steps | 8000000 |
| sample_stride | 20000 |
| checkpoint_steps | 800000 |
| forward_steps | 250000 |
| forward_sample_stride | 5000 |
| temps | 0.40,0.56,0.64,0.72,0.76,0.80,0.84,0.88,0.96 |
| Psi_scale_pi | 0.85 |
| proposal_id | 2 |
| proposal_delta_rad | 0.1 |
| window_radius | 1 |
| init | ordered |
| seed | 1234 |
| forward_seed | 987654 |
| threads_per_block | 128 |
| tap_lags | 1,2,3 |
| tap_weights | 0.5,0.3,0.2 |
| beta_pi | 0.5 |
| rho | 0.1 |
| lambda_safety_factor | 3.0 |
| null_scale | 0.49 |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | None |

### Core Metrics

| Metric | Value |
| --- | --- |
| raw_mae_vs_author | 0.19473891 |
| insertable_mae_vs_author | 0.04179999 |
| insertable_max_abs_error | 0.10275687 |
| insertable_bias | -0.04163009 |
| insertable_rmse | 0.05291059 |
| K1_max_abs_error | 5.00370755e-16 |
| K1_passed | True |
| selector_is_edge | False |
| selector_score | 0.02882443 |
| K2_live_total_kicks | 449 |
| K2_live_kicked_trajectories | 56 |
| K2_live_max_abs_kick_float | 1.50000000 |
| K2_null_total_kicks | 0 |
| live_post_hamming | 495 |

### Causal Insertion Decay

| Field | Value |
| --- | --- |
| stage | causal_insertion_decay_v7 |
| enabled | True |
| uses_author_data | False |
| decay_formula | fit log(coherence_lag)=a-rate*lag across causal taps; rate_excess=max(0, rate(T)-min_T rate(T)); lag_slope_decay=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints); thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max), with T_guide=min(T_ins, simulated chi shoulder); thermal_envelope_factor=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints*(thermal_gain(T)-1)); D_insert(T)=min(1, lag_slope_decay*thermal_envelope_factor); auto horizon=max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps; Psi_Y still uses coherence-thinned taps for K2 freeze; finite-clock insertion_threshold_survival is recorded as a K2 readiness diagnostic, not multiplied into the observable; c_mean_m_insertable=c_mean_m_raw*D_insert(T); lag_strength and default kappa are frozen global finite-horizon calibration constants |
| lag_decay_strength | 0.825 |
| thermal_guide_strength | 0.925 |
| calibration_note | single frozen global finite-horizon calibration; no runtime author-data lookup |
| guide_center_T | 0.76 |
| guide_center_mode | min(T_ins, chi_shoulder) |
| chi_shoulder_T | 0.84 |
| guide_iteration | 1 |
| guide_fixed_point_converged | True |
| guide_width | 0.18666666666666665 |
| guide_strength_mode | frozen_global_default |
| horizon_mode | auto |
| horizon_checkpoints | 5.0125 |
| rate_floor_rule | minimum fitted lag-coherence decay rate across the simulated temperature grid |
| threshold_role | diagnostic/readiness gate for K2 insertion, not an observable multiplier |

### Scout Selector Rows

| T | score | |Psi_g| | chi | c_mean_m | disorder | decay |
| --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.00000000 | 0.31438981 | 0.16137097 | 0.90121915 | 0.09878085 | 1.00000000 |
| 0.560 | 0.01673220 | 0.26412558 | 0.74497344 | 0.83316259 | 0.16683741 | 0.95561761 |
| 0.640 | 0.02325036 | 0.20248731 | 1.00823257 | 0.78741927 | 0.21258073 | 0.88041271 |
| 0.720 | 0.01927854 | 0.16741129 | 1.02792844 | 0.73436939 | 0.26563061 | 0.74450407 |
| 0.760 | 0.02882443 | 0.15702335 | 1.64725299 | 0.70080378 | 0.29919622 | 0.64109349 |
| 0.800 | 0.02005083 | 0.13439035 | 1.95071772 | 0.66258897 | 0.33741103 | 0.52333998 |
| 0.840 | 0.00869801 | 0.11584579 | 2.25454153 | 0.62644305 | 0.37355695 | 0.38247692 |
| 0.880 | 0.00390156 | 0.10776954 | 2.77992419 | 0.58909147 | 0.41090853 | 0.26996679 |
| 0.960 | 0.00000000 | 0.10123604 | 4.28372073 | 0.48071185 | 0.51928815 | 0.11735599 |

### Temperature Curve Rows

| T | |Psi_g| | chi | raw M | insertable M | author M | raw err | insert err | decay | lag decay | thermal env | guide gain | excess rate | pair UPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.314390 | 0.161371 | 0.901219 | 0.901219 | 0.900455 | 7.645862e-04 | 7.645862e-04 | 1.000000 | 1.000000 | 1.000000 | 0.250000 | 0.000000 | 4.583e+06 |
| 0.560 | 0.264126 | 0.744973 | 0.833163 | 0.796185 | 0.827175 | 0.005988 | -0.030990 | 0.955618 | 0.884877 | 1.079943 | 0.371179 | 0.029576 | 4.625e+06 |
| 0.640 | 0.202487 | 1.008233 | 0.787419 | 0.693254 | 0.763384 | 0.024036 | -0.070130 | 0.880413 | 0.793872 | 1.109011 | 0.551760 | 0.055820 | 4.686e+06 |
| 0.720 | 0.167411 | 1.027928 | 0.734369 | 0.546741 | 0.649498 | 0.084872 | -0.102757 | 0.744504 | 0.697874 | 1.066817 | 0.820194 | 0.086986 | 4.712e+06 |
| 0.760 | 0.157023 | 1.647253 | 0.700804 | 0.449281 | 0.522731 | 0.178073 | -0.073450 | 0.641093 | 0.641093 | 1.000000 | 1.000000 | 0.107508 | 4.732e+06 |
| 0.800 | 0.134390 | 1.950718 | 0.662589 | 0.346759 | 0.396152 | 0.266437 | -0.049392 | 0.523340 | 0.587961 | 0.890094 | 1.219224 | 0.128429 | 4.757e+06 |
| 0.840 | 0.115846 | 2.254542 | 0.626443 | 0.239600 | 0.260962 | 0.365481 | -0.021362 | 0.382477 | 0.523855 | 0.730120 | 1.486506 | 0.156346 | 4.775e+06 |
| 0.880 | 0.107770 | 2.779924 | 0.589091 | 0.159035 | 0.165904 | 0.423187 | -0.006869 | 0.269967 | 0.485534 | 0.556020 | 1.812384 | 0.174716 | 4.784e+06 |
| 0.960 | 0.101236 | 4.283721 | 0.480712 | 0.056414 | 0.076899 | 0.403812 | -0.020485 | 0.117356 | 0.451461 | 0.259947 | 2.694119 | 0.192311 | 4.802e+06 |

### Output Artifacts

| Artifact | Path |
| --- | --- |
| anchor_evidence_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_anchor_evidence_20260621_123747.csv |
| checkpoint_manifest_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_checkpoint_manifest_20260621_123747.csv |
| dynamic_trajectory_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_dynamic_trajectory_20260621_123747.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_evidence_20260621_123747.json |
| freeze_manifest_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_freeze_manifest_20260621_123747.json |
| live_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_live_insertion_rows_20260621_123747.csv |
| null_insertion_rows_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_null_insertion_rows_20260621_123747.csv |
| out_dir | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747 |
| projection_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\projection_base.npz |
| smallN_base_npz | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\base_smallN_L2_q8.npz |
| temp_curve_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_temp_curve_20260621_123747.csv |
| trajectory_summary_csv | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_trajectory_summary_20260621_123747.csv |
| evidence_json | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\unified_main_pipeline\20260621_123747\unified_evidence_20260621_123747.json |
| runner_log | C:\clever_paper\analysis\unified_test_series\series_20260621_121023\09_L32_deep\09_L32_deep.log |

### Freeze And Dynamic Outputs

| Field | Value |
| --- | --- |
| freeze_passed | True |
| Pi_lambda0 | 6.277781558097409 |
| Pi_anchor_count | None |
| Pi_candidate_count | None |
| causal_tap_lags | None |
| causal_tap_weights | None |
| resolution_gate_passed | True |
| dynamic_gates | {"dynamic_diverged": true, "live_post_state_hamming_final_total": 495, "max_abs_live_minus_post_final": 0.0004202977869287583, "mean_live_minus_post_final": -8.009743572954253e-06, "null_matches_post_final": true, "null_post_state_hamming_final_total": 0} |
| live_summary | {"kicked_trajectories": 56, "lambda0": 6.277781558097409, "max_abs_kick_float": 1.4999999999999998, "total_kicks": 449} |
| null_summary | {"kicked_trajectories": 0, "lambda0": 1.025370987822577, "max_abs_kick_float": 0.245, "total_kicks": 0} |
| live_post_hamming | 495 |
