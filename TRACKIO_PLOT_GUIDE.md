# UrbanNest Trackio Plot Explanation Guide

## Purpose
This guide explains every plot/metric series logged to Trackio in the notebook, what it means, and how to interpret it correctly.

## 1) Model Comparison Plots
Tracked metrics:
- `search_best_cv_mae_grid`, `search_best_cv_mae_random`, `search_best_cv_mae_bayes`
- `test_mae_grid`, `test_mae_random`, `test_mae_bayes`
- `test_r2_grid`, `test_r2_random`, `test_r2_bayes`
- `search_time_seconds_grid`, `search_time_seconds_random`, `search_time_seconds_bayes`
- `search_trials_count_grid`, `search_trials_count_random`, `search_trials_count_bayes`

What it shows:
- Side-by-side comparison of Grid Search, Random Search, and Bayesian Optimization.

How to interpret:
- MAE: lower is better.
- R²: higher is better.
- Time: lower means cheaper/faster.
- Prefer method with best error-quality balance, not just best single metric.

## 2) Time vs Performance Tradeoff Plot
Tracked metrics:
- `tradeoff_time_seconds`
- `tradeoff_test_mae`
- `tradeoff_test_r2`
(Logged with steps representing methods.)

What it shows:
- Runtime against accuracy quality for each method.

How to interpret:
- Best tradeoff point = low runtime + low MAE (or high R²).
- Useful when slightly better MAE comes with much higher runtime.

## 3) Convergence / Optimization History (All Methods)
Tracked metrics:
- `cmp_best_mae_grid`, `cmp_best_mae_random`, `cmp_best_mae_bayes`
- `cmp_trial_mae_grid`, `cmp_trial_mae_random`, `cmp_trial_mae_bayes`

What it shows:
- Search progress across trials for each method.

How to interpret:
- `cmp_best_mae_*`: best-so-far curve should go down or stay flat.
- `cmp_trial_mae_*`: noisy trial values reveal exploration behavior.
- Faster early drop = better search efficiency under fixed budget.

## 4) Optuna-Specific History Plot
Tracked metrics:
- `optuna_history_best_mae`
- `optuna_history_trial_mae`

What it shows:
- Bayesian search progress only (Optuna).

How to interpret:
- Quick decrease early means TPE found good region fast.
- Flat segment means no new better region discovered.

## 5) Optuna Parameter Importance Plot
Tracked metrics:
- `optuna_param_importance`
- `optuna_param_importance_<param_name>`

What it shows:
- Relative impact of each hyperparameter on objective (MAE).

How to interpret:
- Larger importance = stronger impact on final performance.
- Use this to narrow future search ranges for key params.

## 6) Actual vs Predicted Plot
Tracked metrics:
- `actual_vs_pred_actual`
- `actual_vs_pred_predicted`

What it shows:
- Relationship between ground truth and model predictions.

How to interpret:
- Ideal points lie near 45-degree line.
- Systematic offset suggests consistent over/under prediction.

## 7) Residual Plot
Tracked metrics:
- `residual_plot_predicted`
- `residual_plot_residual`

What it shows:
- Error (`actual - predicted`) against predicted value.

How to interpret:
- Good pattern: random cloud around zero.
- Bad pattern: trend/curve/funnel indicates bias or variance issues.

## 8) Feature Importance Plot (Final Random Forest)
Tracked metrics:
- `rf_feature_importance`
- `rf_feature_importance_<feature_name>`

What it shows:
- Top features contributing to predictions.

How to interpret:
- Higher value = stronger model reliance.
- Validate whether important features match domain intuition.

## 9) Cross-Validation Fold Stability Plots
Tracked metrics:
- `cv_fold_mae_grid`, `cv_fold_mae_random`, `cv_fold_mae_bayes`
- `cv_fold_r2_grid`, `cv_fold_r2_random`, `cv_fold_r2_bayes`
- `cv_mean_mae_*`, `cv_std_mae_*`, `cv_mean_r2_*`, `cv_std_r2_*`

What it shows:
- Fold-wise consistency across methods.

How to interpret:
- Lower std = more stable and reliable method.
- High variance across folds = sensitive to data split.

## 10) Final Validation Metrics
Tracked metrics:
- `final_model_test_mae`
- `selected_method_matches_best`

What it shows:
- Final deployed model performance and selection consistency.

How to interpret:
- `selected_method_matches_best = 1` confirms selected method aligns with best CV result.

## Common Points
1. MAE was primary optimization metric due to rent-scale interpretability.
2. Convergence curves show search efficiency under equal trial budget.
3. Tradeoff plot justifies method choice beyond pure error.
4. Residual and actual-vs-predicted plots validate predictive behavior.
5. Fold variance confirms robustness, not just average performance.
