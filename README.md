# Numerical Optimization Benchmarks

## 📌 Project Overview
This repository features an in-depth study of **Numerical Optimization** algorithms. It focuses on comparing iterative vs. recursive logic and quadratic interpolation methods to solve non-linear optimization problems in engineering and AI.

## 🛠 Implemented Algorithms
- **Golden Section Search (Iterative & Recursive):** A robust unimodal interval reduction method based on the Golden Ratio ($\phi \approx 1.618$).
- **Parabolic Interpolation:** A high-order approximation method that utilizes quadratic fitting to find local minima in smooth functions.

## 📊 Quantitative Benchmark: Database Performance Tuning
We analyzed a database query delay function $L(c)$ to determine the optimal cache size ($c$):
$$L(c) = 0.002(c - 120)^2 + 8 + \frac{400}{c + 20}$$

### Comparative Results:
| Parameter | Iterative Solution | Recursive Solution | Precision Difference |
| :--- | :--- | :--- | :--- |
| **Optimal Cache ($c^*$)** | **124.771283** | **124.771283** | $2.48 \times 10^{-7}$ |
| **Minimum Delay $L(c^*)$** | **10.808509** | **10.808509** | $0.00$ |

**Analysis:** The results demonstrate that both recursive and iterative implementations converge to the exact same optimal point. For real-time database systems, the iterative approach is generally preferred to minimize stack overhead.

## 🤖 AI Learning Rate Optimization
Using **Parabolic Interpolation**, the optimal learning rate ($\eta$) for a specific loss function was identified:
- **Computed Optimal $\eta^*$:** $\approx 0.338$

## 🚀 Environment & Setup
- **Language:** Python
- **Dependencies:** `NumPy` (for matrix and mathematical operations)
- **Usage:** Open `test5-1.ipynb` to view the full execution trace and benchmark logic.

## 🎓 Academic Contribution
This repository was established as part of the Master's program in **Computer Engineering** at **Bilecik Şeyh Edebali University**. It highlights the practical application of calculus and numerical methods in software system optimization.
