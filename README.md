# Continuous-time-portfolio-selection-using-Reinforcement-Learning-and-Stochastic-control-

This project implements a **reinforcement learning (RL) framework** for continuous-time **mean-variance (MV) portfolio selection**, where both the portfolio allocations (actions) and wealth (states) are treated as continuous variables. This approach is particularly suitable for **ultra-high frequency trading** and leveraging the large volumes of tick data available in modern electronic markets.

---

## **1. Background**

The classical continuous-time MV portfolio model is a special case of a **stochastic linear-quadratic (LQ) control problem** (Zhou and Li, 2000). Wang et al. (2019) extended this with an **entropy-regularized, exploratory stochastic control formulation**, explicitly balancing **exploration** and **exploitation** in RL. In the infinite horizon case, the optimal exploratory control distributions are Gaussian, providing theoretical justification for Gaussian exploration commonly used in RL.

---

## **2. Contributions**

This project focuses on the **finite-time horizon MV problem**, extending the exploratory LQ framework:

- **Global Optimal Solution:** Derives the optimal Gaussian feedback control with **time-decaying variance**, indicating that exploration naturally decreases as the investment horizon approaches its end.  
- **Separation of Exploration and Exploitation:** The mean of the Gaussian distribution drives exploitation, while the variance captures exploration.  
- **Effect of Random Environments:** Shows how environmental randomness positively influences learning.  
- **Interpretability and Implementability:** Designs an RL algorithm based on a **policy improvement theorem** for continuous-time stochastic control, reducing general non-parametric policies to a **parametrized Gaussian family**, ensuring **fast convergence** to the global optimum.  

---

## **3. Methodology**

- Initialize a Gaussian policy with chosen mean and variance.
- Iteratively update the policy based on the **value function** of the current policy.
- Exploit the **explicit parametric form** to separate exploration and exploitation.
- Compare performance against: **Adaptive control** using real-time maximum likelihood parameter estimation.

---

## **4. Implementation**

- The algorithm is tested on both **simulated market scenarios** with stationary and non-stationary investment opportunities, as well as on real data.  
- Performance metrics include **portfolio return** and **variance**.  

---

## **5. References**

- Zhou, X. Y., & Li, D. (2000). *Continuous-Time Mean-Variance Portfolio Selection: A Stochastic LQ Framework*.  
