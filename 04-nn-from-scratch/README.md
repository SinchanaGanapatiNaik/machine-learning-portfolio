# Neural Network From Scratch (NumPy)

Reimplements forward propagation using only NumPy — no TensorFlow/Keras — to verify a genuine understanding of what happens mathematically inside a trained neural network, rather than just knowing how to call `.predict()`.

## Approach

1. Reuse the exact same trained model from [Project 3 (Purchase Prediction)](../03-purchase-prediction) — same data, same architecture, same trained weights.
2. Extract the model's 531 trained parameters via `model.get_weights()` (3 weight matrices + 3 bias vectors, one pair per layer).
3. Implement a single dense layer by hand: `Z = X @ W + b`, followed by manual sigmoid activation.
4. Chain three of these layers together into a complete forward-pass function.
5. Validate the from-scratch implementation against Keras's own `model.predict()` — first on a single customer, then across the full test set.

## Verification

| Check | Result |
|---|---|
| Single customer prediction match | Identical to 8 decimal places (0.08104385) |
| Max difference across all 100 test customers | 7.62 × 10⁻⁸ (floating-point noise) |
| Accuracy using from-scratch predictions | 0.74 — identical to Keras |

## Key takeaway

The from-scratch implementation doesn't just produce numerically similar output — it is mathematically identical to Keras's internal computation, down to floating-point precision, and makes the exact same real-world purchase/no-purchase decision for every test customer. This confirms a working understanding of forward propagation mechanics (matrix multiplication, bias addition, activation functions, layer chaining), rather than reliance on a high-level framework abstracting those details away.

## Tech stack

`numpy`, `tensorflow`/`keras` (for the reference model), `scikit-learn`, `pandas`
