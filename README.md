# Beyond Accuracy: Enhancing Parkinson’s Diagnosis with Uncertainty Quantification of Machine Learning Models

## Abstract

Machine learning and deep learning models have shown significant potential in medical diagnosis, but their reliability is crucial for clinical applications. This study investigates uncertainty quantification methods to enhance model trustworthiness in Parkinson’s disease diagnosis. We evaluate Monte Carlo Dropout, Deep Evidential Classification, and Bayesian Neural Networks on three datasets capturing finger tapping, facial expressions, and speech patterns. Results show that Deep Evidential Classification underperforms in both classification accuracy and uncertainty estimation, while Monte Carlo Dropout and Bayesian Neural Networks demonstrate superior reliability. Incorporating uncertainty quantification helps identify uncertain predictions, reducing misdiagnoses and fostering safer AI integration in healthcare.

## Hyperparameter Tuning

This section summarizes the hyperparameter search spaces used for uncertainty estimation methods: **MC Dropout**, **Deep Evidential Classification (DEC)**, and **Bayesian Neural Networks (BNNs)**. The tables below outline the tunable parameters, their ranges or options, and the sampling distributions.

### MC Dropout Hyperparameters

| Hyperparameter        | Values / Range             | Distribution     |
| --------------------- | -------------------------- | ---------------- |
| Model Type            | `ann`, `shallow_ann`       | Categorical      |
| Seed                  | Fixed set                  | Categorical      |
| Learning Rate (`lr`)  | `1e-4` to `1e-1`           | Log-uniform      |
| Max Epochs            | `5` to `100`               | Uniform          |
| Dropout Probability   | `0.1` to `0.5`             | Uniform          |
| Num. Estimators       | `100` to `1000` (step=100) | Discrete Uniform |
| Correlation Threshold | `0.0` to `1.0`             | Uniform          |
| Feature Scaler        | `standard`, `minmax`       | Categorical      |
| Optimizer             | `sgd`, `adamw`             | Categorical      |
| Momentum (SGD only)   | `0.5` to `0.99`            | Uniform          |
| Weight Decay          | `0.0` to `0.1`             | Uniform          |
| Beta₁ (AdamW)         | `0.8` to `0.99`            | Uniform          |
| Beta₂ (AdamW)         | `0.9` to `0.999`           | Uniform          |

---

### DEC-Specific Hyperparameter

| Hyperparameter                       | Values / Range   | Distribution |
| ------------------------------------ | ---------------- | ------------ |
| Regularization Weight (`reg_weight`) | `1e-5` to `1e-3` | Log-uniform  |

---

### BNN-Specific Hyperparameters

| Hyperparameter                         | Values / Range       | Distribution     |
| -------------------------------------- | -------------------- | ---------------- |
| Model Type                             | `bnn`, `shallow_bnn` | Categorical      |
| KL Divergence Weight (`kl_weight`)     | `1e-4` to `1e-1`     | Log-uniform      |
| Num. Posterior Samples (`num_samples`) | `1` to `10`          | Discrete Uniform |

## Contact

For any queries contact,

- Asif Azad - asifazad0178@gmail.com
