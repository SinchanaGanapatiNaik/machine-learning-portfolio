# Purchase Prediction — Neural Network

Predicts whether a customer will make a purchase based on four behavioral and demographic features, using a small feedforward neural network built in TensorFlow/Keras. Includes a comparison against a naive baseline and logistic regression to evaluate whether the added complexity of a neural network is actually justified for this problem.

## Dataset

Synthetic data generated for this project (500 samples), not sourced from a real-world dataset. Features were sampled from realistic distributions:

| Feature | Description | Distribution |
|---|---|---|
| `age` | Customer age | Normal(35, 10), clipped to [18, 70] |
| `income` | Annual income | Normal(50000, 15000), clipped to [10000, 150000] |
| `visits` | Website visits per month | Poisson(5) |
| `session_duration` | Average session length (minutes) | Normal(8, 3), clipped to [0.5, 30] |

The target label (`purchased`) was generated from a designed scoring function that includes a deliberate **interaction effect**: income only influences purchase probability in combination with visit frequency (i.e., high income alone or high visits alone contribute little — both together matter). This was intentional, since interaction effects are a pattern linear models like logistic regression structurally cannot represent, while neural networks can.

Labels were sampled probabilistically (not thresholded), so the dataset includes realistic noise rather than a perfectly separable pattern.

## Approach

1. Generate synthetic data with a known ground-truth relationship (including one non-linear interaction term).
2. Split into train/test sets (80/20), with feature scaling (`StandardScaler`) fit only on the training set to avoid data leakage.
3. Train a 3-layer feedforward network (25 → 15 → 1 units, sigmoid activations, binary crossentropy loss, Adam optimizer).
4. Compare against two baselines:
   - **Naive baseline** — always predicting the majority class.
   - **Logistic regression** — a simple linear model trained on the same scaled data.

## Results

| Model | Test Accuracy |
|---|---|
| Naive baseline (always predict no purchase) | 0.71 |
| Logistic Regression | **0.75** |
| Neural Network (400 epochs) | 0.74 |

## Key takeaway

Even with a genuine multiplicative interaction built into the data — something logistic regression cannot represent directly — the neural network only matched, and slightly trailed, logistic regression's performance.

Likely explanations:
- **Sigmoid activations** are known to slow gradient-based learning in multi-layer networks (the vanishing gradient problem), a limitation this course addresses next via ReLU. This project intentionally used only sigmoid, consistent with the material covered at this stage.
- **Limited training data** (400 examples) may be insufficient for a 531-parameter network to reliably learn a subtle interaction, whereas logistic regression's much smaller parameter count needs far less data to fit well.

This project's conclusion isn't "neural networks are better" — it's that **a neural network's added flexibility only pays off when it has the right architecture and enough data to exploit it.** Reaching for a neural network by default, without checking a simpler baseline, would have hidden this finding entirely.

## What's next

A follow-up (post-Activation-Functions material) will re-run this same setup with ReLU hidden-layer activations to test whether that alone closes the gap.

## Tech stack

`numpy`, `pandas`, `scikit-learn`, `tensorflow`/`keras`, `matplotlib`
