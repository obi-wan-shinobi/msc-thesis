# Experiment EXP001 — Finite-width MLPs converge to NTK limit

**Date:** 2025-10-17  
**Script:** ntk-experiment.ipynb  
**Paper(s):** Neural Tangent Kernel: Convergence and Generalization in Neural Networks (https://arxiv.org/abs/1806.07572)

---

## 1 Goal

- Reproduce results from NTK paper for kernel profile showing low variance for kernel as width increases

![](plots/ntk-profile/ntk-paper-profile.png)

- Test if finite-width MLPs converge to NTK limit

---

## 2 Experiments

### 2.1 Reproduction of NTK kernel profile:

#### Setup:

- **Probe manifold: unit circle (2D).**  
  Use angles $\gamma \in [-\pi, \pi]$ and define

  $$
    x(\gamma)=\begin{bmatrix}\cos\gamma\\ \sin\gamma\end{bmatrix}\in\mathbb{R}^2.
  $$

  Build a dense grid $\Gamma=\{\gamma_i\}_{i=1}^{N}$ (e.g., $N=360$) and the probe set $X_{\text{circle}}=\{x(\gamma_i)\}$.
  Fix the **anchor** $x_0=(1,0)$.

- **Regression task used for training (separate from the probe).**  
  Draw inputs $x\sim\mathcal N(0,I_2)$ and set the target

  $$
    f^*(x)=x_1x_2.
  $$

  Train with mean-squared error on this Gaussian dataset. The circle is **only** for measuring/plotting the kernel.

  ![](plots/ntk-profile/gaussian_vs_circle.png)

- **Model (finite net).**  
  Fully-connected ReLU MLP in **NTK parameterization** with **depth \(L=4\)**
  Compare **two widths** $n\in\{500,\,2048\}$.

- **Training protocol.**  
  Full-batch gradient descent on the Gaussian regression for **200 steps** with **learning rate 1.0**.  
  Repeat for **10 random initializations** per width (to visualize variability across seeds).

- **What is plotted.**  
  The **empirical NTK** profile

  $$
    \gamma \;\mapsto\; \Theta^{(4)}_{\theta_t}\!\big(x_0,\;x(\gamma)\big)
  $$

  measured on the **unit circle** at two times:

  - **$t=0$** (initialization; solid lines),
  - **$t=200$** (after training; dotted lines).  
    Plot **all seeds** as thin curves for each width.

- **Expected observations.**  
  Variance across seeds **shrinks** as width increases (kernel concentrates to a deterministic limit).

#### Results

![](plots/ntk-profile/ntk-profile.png)

This serves as a sanity check for the code as well as the idea that as width starts increasing,
the kernel shows less variance eventually becomes constant as $width \to \infty$.

**Observations**  
 The empirical NTK profile $\gamma \mapsto \Theta(x_0,x(\gamma))$ matches the paper's qualitative behavior:
variance across random initializations **shrinks with width**, and the mild post-training "inflation" diminishes as width grows.

_Takeaway:_ the kernel concentrates toward a deterministic limit and is nearly constant during training when wide.

---

### 2.2 Function-space convergence on the unit circle

- **Inputs on the unit circle (2D).**  
  For angles $\gamma \in [-\pi, \pi)$,

  $$
    x(\gamma) = \begin{bmatrix}\cos\gamma \\ \sin\gamma\end{bmatrix} \in \mathbb{R}^2.
  $$

  Construct a dense grid $\Gamma = \{\gamma_i\}_{i=1}^N$ and the corresponding dataset $X_{\mathrm{eval}} = \{x(\gamma_i)\}_{i=1}^N$.

- **Targets**  
  **Simple baseline (paper-like):** $f^*(x)=x_1x_2$. On the circle this equals $\tfrac{1}{2}\sin(2\gamma)$ (a single low-frequency mode).  
  ![](plots/simple-regression/simple-regression-task.png)

- **Train/Test split on the circle.**  
  Select a **small, random** training subset $X_T=\{x(\gamma_{i_j})\}_{j=1}^{M}$ from $X_{\mathrm{eval}}$ (e.g., $M \in \{64, 256\}$).  
  Targets are $y_T = \{y(\gamma_{i_j})\}$.  
  Evaluate on the full grid $X_{\mathrm{eval}}$.

- **Model (finite net).**  
  Fully-connected ReLU MLP in **NTK parameterization**, depth $L = 1$ (Can try with $L=4$ but larger widths run out of memory), width $n$ swept over
  $\{64,128,256,512,1024,2048,4096,8192\}$.  
  **Loss:** MSE. **Optimizer:** full-batch GD/SGD with a small learning rate (e.g., $10^{-2}$) to approximate gradient flow; train to near-zero train MSE.

- **Analytic NTK (infinite width).**  
  Build the deterministic limit kernel using the **same architecture/activation/parameterization and init hyperparameters**.  
  **NTK predictor (kernel ridge):**

  $$
    \alpha=(K_{TT}+\lambda I)^{-1} y_T,\qquad
    \hat y_{\infty}(X)=K_{XT}\,\alpha,
  $$

  with small $\lambda$ (e.g., $10^{-6}$). $K_{TT}$ is the NTK gram matrix between the training points, $K_{XT}$ is the NTK gram matrix between the test points and the training points.

- **Across-seed robustness.**  
  For each width, repeat training over multiple random initializations (e.g., 10 seeds); summarize by median and percentile bands.  
  _(Seeds affect finite nets; not the analytic NTK limit.)_

### 2.3 Function-space convergence on a harder Fourier mixture.

$$
  y(\gamma) = \sum_{k\in\mathcal K} a_k\,\sin\!\big(k\,\gamma+\phi_k\big) + \varepsilon,\quad
  \varepsilon \sim \mathcal N(0,\sigma^2),
$$

with $\mathcal K$ including higher frequencies (e.g., $\{2,4,7,11,16,23,32\}$), mildly decaying amplitudes $a_k$, and phases $\phi_k \in \{0.0, 0.5\pi, 1.2, 2.0, 0.3\pi, 4.5, 5.8\}$(arbitrarily chosen).

![](plots/complex-regression/complex-regression-task.png)

### Results

**Comparison in function space:**

- $\hat y_n(\gamma)$: Predictions of $n$-width network.
- Overlay $\hat y_n(\gamma)$ and $\hat y_{\infty}(\gamma)$ on $\gamma\in[-\pi,\pi)$.
- Record

  $$
    \mathrm{RelErr}(n)=\frac{\|\hat y_n-\hat y_{\infty}\|_2}{\|\hat y_{\infty}\|_2}.
  $$

- **Convergence:** As width $n$ increases, $\hat y_n \to \hat y_{\infty}$ and RelErr$(n)$ decreases.
- **Task difficulty:** The simple baseline $x_1x_2=\sin(\gamma)\cos(\gamma)=\tfrac12\sin(2\gamma)$ is **low-frequency** and easy; many widths will already match the NTK closely. The **harder mixture** (with higher modes and fewer train points) should make differences visible at small widths and highlights convergence as $n\uparrow$.
- **Optimization:** Full-batch GD best mirrors NTK gradient flow.

#### 2.2 Simple regression task

![Relative Error on simple regression task](plots/simple-regression/rel-err-on-simple-regression-task.png)
_Fig: Relative Error on simple regression task_

![Convergence on simple regression task](plots/simple-regression/convergence-on-simple-regression-task.png)
_Fig: Convergence on simple regression task_

### 2.3 Complex regression task

![Relative Error on complex regression task](plots/complex-regression/rel-err-on-complex-regression-task.png)
_Fig: Relative Error on complex regression task_

![Convergence on complex regression task](plots/complex-regression/convergence-on-complex-regression-task.png)
_Fig: Convergence on complex regression task_

![Convergence on partial fourier mixture](plots/complex-regression/convergence-on-partial-complex-regression-task.png)
_Fig: Convergence on complex regression task (partial Fourier mixture)_

### Observations

- **Simple regression task ( $f^*(x)=x_1x_2 = \frac{1}{2} \sin 2\gamma$ on $S^1$).**  
  The **relative error (RelErr)** between finite nets and the analytic NTK predictor is **roughly flat** across widths (minor improvements only).  
  _Interpretation:_ this target is dominated by a **single low-frequency eigenmode** of the NTK on the circle, which small nets already capture well; finite-width corrections are tiny, so width yields little visible gain.

- **Complex regression task (Fourier mixture with higher modes).**  
  The **RelErr decreases** monotonically (or near-monotonically) as width increases, **but the absolute values remain extremely small** — e.g., from roughly 0.60 to 0.55 — which makes the improvement only marginal in absolute terms.  
  _Interpretation:_ although there is a weak downward trend, the finite-width networks all appear to converge to **similar solutions that differ from the NTK predictor**, suggesting that they are not fully reaching the infinite-width limit but are only able to approximate the lower harmonics, as can be seen in the last two plots above.

- **Overlay plots (function space).**  
  For both tasks, the **finite-net predictions** look visually close for all widths; differences are subtle by eye.

### Possible reasons the simple task shows little RelErr improvement

- The target is **too easy/low-rank** for the NTK on $S^1$; even narrow nets approximate the leading eigenfunction well.

### 2.4 Same complex task with low widths

The target is a controlled Fourier mixture with only 7 harmonics,

$$
y(\gamma) = \sum_{k\in\{2,4,7,11,16,23,32\}} a_k sin(k\gamma+\phi_k)
$$

Since the input features are $(\sin\gamma, \cos\gamma)$, a ReLU MLP can synthesize low-order trigonometric polynomials with few units; hence all widths look similar. To expose visible progression with width, we sweep very small widths (2-10, 100) and compare to the infinite-width NTK predictor.

![Relative Error with low widths on complex regression task](plots/low-width-complex-regression/rel-err-low-widths-on-complex-regression-task.png)
_Fig: Relative Error for lower widths on the complex regression task_

![Convergence using lower widths on complex regression task](plots/low-width-complex-regression/convergence-low-widths-on-complex-regression-task.png)
_Fig: Convergence using lower widths on complex regression task_

![Convergence on partial fourier mixture](plots/low-width-complex-regression/convergence-low-widths-on-partial-complex-regression-task.png)
_Fig: Convergence on complex regression task (partial Fourier mixture)_

### 2.5 Same complex task with low widths + some high-widths but with higher learning rate and more gradient descent steps

It seems that the finite width networks converge only for low frequency modes.
According to the spectral bias paper (https://arxiv.org/abs/1806.08734) it might just be the case that high-frequency modes need more convergence time, so we need more gradient descent steps and higher learning rate (larger $\eta t$).

Updated setup:

- width $w \in \{2,4,6,8,10,100,1000,10000\}$
- learning rate $\eta = 1$ (was 0.01 earlier)
- timesteps $100,000$

![Relative Error with higher LR and more GD steps on complex regression task](plots/low-width-high-lr-timesteps/rel-err-low-widths-more-steps-on-simpler-regression-task.png)
_Fig: Relative Error for lower widths with high LR and more GD steps on the complex regression task_

![Convergence with higher LR and more GD steps on complex regression task](plots/low-width-high-lr-timesteps/convergence-low-widths-more-steps-on-simpler-regression-task.png)
_Fig: Convergence using lower widths with high LR and more GD steps on complex regression task_

![Convergence for last large widths](plots/low-width-high-lr-timesteps/convergence-last-low-widths-more-steps-on-simpler-regression-task.png)
_Fig: Cleaner picture of convergence for width 10, 100, 1000, 10000 on complex regression task_

---

## 6 Outcome

- **Validated NTK regime qualitatively:** kernel profiles reproduce the **concentration** ($\downarrow$ variance with width) and **near constancy** during training reported in the NTK paper.
- **Function-space convergence:** on a **harder target** with higher Fourier content, the **RelErr** decreases with width, supporting convergence to the NTK predictor but with high learning rate and more gradient descent steps, this doesn't seem to be faithful to the gradient flow dynamics since the step sizes are large, we need to understand this more.

## 7 Next Steps

- Verify the weak-learning regime theory from Hanin-Nica (https://arxiv.org/pdf/1909.05989)
