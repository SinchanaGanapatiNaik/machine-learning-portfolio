# Machine Learning Portfolio
A running collection of ML/AI projects built while working through my learning roadmap — starting from classical ML fundamentals and building up toward deep learning, LLMs, and agentic AI systems.
## Projects
| # | Project | Techniques | Results |
|---|---|---|---|
| 01 | [House Price Prediction](./01-housing-regression) | Linear/Ridge regression, IQR outlier capping, one-hot encoding, from-scratch gradient descent | R² 0.700 (from-scratch), R² 0.684 (sklearn) |
| 02 | [Loan Approval Prediction](./02-loan-approval) | Logistic regression, feature scaling (StandardScaler), one-hot encoding, coefficient interpretation | Accuracy 0.905, Recall (Rejected) 0.86 (scaled) vs 0.61 (unscaled) |
| 03 | [Purchase Prediction](./03-purchase-prediction) | Feedforward neural network (TensorFlow/Keras), synthetic data with engineered interaction term, baseline comparison vs logistic regression | Accuracy 0.74 (NN) vs 0.75 (logistic regression) vs 0.71 (naive baseline) |
| 04 | [Neural Network From Scratch](./04-nn-from-scratch) | Manual forward propagation in NumPy (matrix multiplication, bias, sigmoid), validated against a trained Keras model | Max difference vs Keras: 7.62e-08; accuracy match: 0.74 (identical) |
*(More projects added as I progress — classification, regularization comparisons, neural networks, and eventually RAG/agents.)*
## Background
Currently working through Andrew Ng's Machine Learning Specialization (Coursera), building a project after key concepts to reinforce them practically rather than just completing course exercises.
