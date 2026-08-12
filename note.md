## Practical recommendation

This guideline might be useful for further collaborator/team members.

1. Start with PLS regression on the engineered per-experiment feature table (as built in Section 2) as an interpretable benchmark, tuning the number of latent components by cross-validated R^2/MAE.

2. Use Gradient Boosting / Random Forest as the primary predictive model, because it captures nonlinearity without overfitting on 100 samples when combined with shallow trees, few boosting rounds/strong shrinkage, and rigorous CV (repeated K-fold given the small N).

3. Tune and select via nested/repeated cross-validation (not a single train/validation split) because with 99 samples a single split has high variance; report CV mean +- std, not a point estimate.

4. Keep the feature set compact and physically meaningful: prefer AUC/final-value summaries over dozens of redundant statistics, or feed the correlated features through PLS/PCA to reduce dimensionality before the tree model if the feature-to-sample ratio becomes a concern.

5. Consider a Gaussian Process: because it is a complementary uncertainty-aware model, 
   
6. Consider a hybrid mechanistic + ML model - it is the most scientific option for a process with known cell-growth kinetics and is the technique that most bioprocess-modeling teams take once black-box performance plateaus.

7. We should use deep learning when we have more experiment data available, or use a small model architecture e.g., a small 1D-CNN/GRU for feature extraction from the time series, feeding into a simple regressor head.

## Model recommendation

| Model | Fit for this dataset | Reasons |
|---|---|---|
| **PLS / PCR / regularized linear regression** | Recommended as the primary/benchmark model | PLS is the standard in bioprocess chemometrics for this setup: small N, collinear regressors, batch time-series unfolded into scalar summaries. It handles multicollinearity natively via latent variables, is highly interpretable (loadings show which process phases/variables drive titer), and is very hard to overfit with only a handful of latent components on 99 samples. |
| **Random Forest / Gradient Boosting / XGBoost** | Recommended as the primary predictive model | Tree ensembles capture the nonlinearities and interactions (e.g., feed rate x duration) that a linear/PLS model misses, remain fairly robust with small N (via bagging/shrinkage + shallow trees + strong CV), are insensitive to collinearity, and directly provide feature importance. |
| **Gaussian Process Regression** | Worth trying as a secondary model | Can be used with small-N regression; gives predictive uncertainty, which is valuable when the model will inform experiment design/process optimization decisions (e.g., "how confident are we in this titer prediction?"). |
| **Deep learning (LSTM/GRU/1D-CNN/Transformer)** | Not recommended given small amount of data volume | With 99 training experiments, a sequence model has far more parameters than data points and will overfit badly. But, it might be used within a *hybrid model* that couples a mechanistic ODE (Monod-type growth/substrate-consumption kinetics) with a small neural correction term. |
