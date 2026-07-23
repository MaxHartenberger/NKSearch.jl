# Gauge Drift in L-BFGS for Relative Periodic Orbits

## 1. Multiple-shooting formulation

A relative periodic orbit (RPO) of period $T$ and spatial shift $s$ is a
root of the multiple-shooting system

$$
F(z) = \begin{pmatrix}
\varphi(T/N, u_1) - u_2 \\
\vdots \\
\varphi(T/N, u_{N-1}) - u_N \\
S\bigl(\varphi(T/N, u_N),\, s\bigr) - u_1 \\[4pt]
0 \\
0
\end{pmatrix} = 0,
\qquad
z = \begin{pmatrix} u_1 \\ \vdots \\ u_N \\ T \\ s \end{pmatrix} \in \mathbb{R}^{nN+2},
$$

where $\varphi$ is the flow map, $S$ is a spatial shift operator, and the
last two components are phase-fixing constraints that remove the two
continuous symmetries (time translation and space translation).  With
reference state $u_{\text{ref}} = u_1^{(0)}$ (the initial $u_1$), these
are

$$
\begin{aligned}
F_{N+1}(z) &= \bigl\langle u_1 - u_{\text{ref}},\; f(u_{\text{ref}}) \bigr\rangle, \\[2pt]
F_{N+2}(z) &= \bigl\langle u_1 - u_{\text{ref}},\; \partial_s S(u_{\text{ref}}, 0) \bigr\rangle,
\end{aligned}
$$

where $f(u) = \dot{u}$ is the vector field.  By construction
$F_{N+1}(z^{(0)}) = F_{N+2}(z^{(0)}) = 0$ at the initial guess.

The augmented Jacobian $J(z) = DF(z) \in \mathbb{R}^{(nN+2) \times (nN+2)}$
has the block-bidiagonal structure

$$
J(z) = \begin{pmatrix}
\Phi_1 & -I &        &        &  & \frac{1}{N}f(\varphi_1) & 0 \\[4pt]
       & \Phi_2 & -I &        &  & \frac{1}{N}f(\varphi_2) & 0 \\[4pt]
       &        & \ddots & \ddots &  & \vdots & \vdots \\[4pt]
       &        &        & \widehat\Phi_N & -I & \frac{1}{N}\widehat f_N & \partial_s S(\varphi_N, s) \\[4pt]
f(u_{\text{ref}})^\top & 0 & \dots & 0 & 0 & 0 & 0 \\[4pt]
\partial_s S(u_{\text{ref}}, 0)^\top & 0 & \dots & 0 & 0 & 0 & 0
\end{pmatrix},
$$

where $\Phi_i = D\varphi(T/N, u_i)$, $\widehat\Phi_N = \partial_u S(\varphi_N, s)\,\Phi_N$,
and $\widehat f_N = \partial_u S(\varphi_N, s)\,f(\varphi_N)$.

---

## 2. The true gauge degeneracy

The **unaugmented** system (rows $1$ through $N$ only) has a continuous
degeneracy: shifting **all** shooting points by a common time offset $\tau$
leaves the continuity equations satisfied to first order.  Writing
$u_i(\tau) = \varphi(\tau, u_i)$, the tangent to this family is

$$
w_{\text{time}} = \bigl(f(u_1),\; f(u_2),\; \dots,\; f(u_N),\; 0,\; 0\bigr).
$$

For a near-RPO the linearised flow approximately maps tangent vectors:
$\Phi_i\,f(u_i) \approx f(u_{i+1})$, so

$$
(Jw_{\text{time}})_i = \Phi_i f(u_i) - f(u_{i+1}) \approx 0 \qquad (i = 1,\dots,N-1).
$$

The closing segment contributes $(Jw_{\text{time}})_N = \widehat\Phi_N f(u_N) - f(u_1)$,
which is also small for a near-RPO.  Likewise a common spatial shift generates

$$
w_{\text{space}} = \bigl(\partial_s S(u_1,0),\; \dots,\; \partial_s S(u_N,0),\; 0,\; 0\bigr).
$$

The direction $w_{\text{time}}$ (and analogously $w_{\text{space}}$) is an
**approximate** null direction of the continuity rows of $J$ — it becomes
an exact null direction in the limit of a converged RPO.  The objective

$$
\phi(z) = \frac{1}{2}\|F(z)\|^2
$$

therefore has small curvature along $w_{\text{time}}$ and $w_{\text{space}}$
when the residual is small.

---

## 3. How Newton methods pin the gauge

The hookstep / trust-region Newton method solves

$$
J(z_k)\,\delta z = F(z_k)
$$

at each iteration.  Even though $F(z_k).d = (0,0)$ (the phase components
are set to zero in the code — see §6), the **phase rows of $J$ actively
constrain the Newton step**.  The row $f(u_{\text{ref}})^\top$ forces

$$
\langle \delta u_1,\; f(u_{\text{ref}}) \rangle = 0,
$$

which, together with the coupling through the continuity rows, prevents
the step from moving along the gauge direction $w_{\text{time}}$.  The
gauge is pinned by the matrix structure of $J$.

---

## 4. How L-BFGS sees the Jacobian

L-BFGS minimises $\phi$ using only gradient information:

$$
\nabla\phi(z) = J(z)^\top F(z).
$$

### 4.1 Phase columns contribute nothing

The phase columns of $J^\top$ (columns $N+1$ and $N+2$) are

$$
[J^\top]_{1,\,N+1} = f(u_{\text{ref}}), \qquad
[J^\top]_{1,\,N+2} = \partial_s S(u_{\text{ref}}, 0),
$$

with zeros elsewhere.  Their contribution to $\nabla\phi$ is

$$
[J^\top]_{*,\,N+1}\,F_{N+1}(z) \;+\; [J^\top]_{*,\,N+2}\,F_{N+2}(z).
$$

In the code, $F_{N+1}$ and $F_{N+2}$ are **explicitly set to zero** at
every residual evaluation (via `b.d = zero.(b.d)`).  Hence the phase
columns of $J^\top$ contribute **nothing** to $\nabla\phi$ at any
iteration.

### 4.2 Continuity columns do contribute gauge components

The **continuity columns** of $J^\top$ (columns $1$ through $N$) multiply
the continuity residuals $F_1,\dots,F_N$, which are nonzero during
optimization.  Computing the component of $\nabla\phi$ along the correct
gauge direction $w_{\text{time}}$:

$$
\begin{aligned}
\langle \nabla\phi(z),\, w_{\text{time}} \rangle
&= \langle F(z),\, J(z)\,w_{\text{time}} \rangle \\[4pt]
&= \sum_{i=1}^{N-1} \bigl\langle F_i,\; \Phi_i f(u_i) - f(u_{i+1}) \bigr\rangle
   \;+\; \bigl\langle F_N,\; \widehat\Phi_N f(u_N) - f(u_1) \bigr\rangle \\[4pt]
&\quad +\; \underbrace{F_{N+1}\,\langle f(u_{\text{ref}}), f(u_1) \rangle}_{=0}
   \;+\; \underbrace{F_{N+2}\,\langle \partial_s S(u_{\text{ref}},0), f(u_1) \rangle}_{=0}.
\end{aligned}
$$

The phase-condition terms vanish because $F_{N+1} = F_{N+2} = 0$.  However,
the **continuity terms are generally nonzero** because $F_i \neq 0$ and
$\Phi_i f(u_i) - f(u_{i+1}) \neq 0$ for an unconverged orbit.

### 4.3 But the continuity contribution is small

For a **near-RPO**, the mismatch $\Phi_i f(u_i) - f(u_{i+1})$ is small —
it vanishes exactly for a converged orbit.  Each term is $O(\|F_i\| \cdot \delta)$
where $\delta$ measures how far the orbit is from satisfying the
tangent-propagation property.  The gauge-direction gradient component
from the continuity rows is therefore weak.

In contrast, the gradient components in the **non-gauge** directions come
from the diagonal blocks $\Phi_i$ and the identity blocks of $J$, which
give $O(\|F_i\|)$ contributions — a full order larger when $\delta$ is
small.

---

## 5. Consequence for L-BFGS

L-BFGS builds a quasi-Newton approximation $H_k \approx (\nabla^2\phi)^{-1}$
from gradient differences

$$
y_k = \nabla\phi(z_{k+1}) - \nabla\phi(z_k).
$$

The $y_k$ vectors have a **weak** component along $w_{\text{time}}$ and
$w_{\text{space}}$ (sourced only from the continuity rows, not from the
phase rows).  The curvature information in the gauge directions that
$H_k$ can accumulate is therefore limited, especially early in the
optimization when the history is shallow.

With two gauge directions (RPO, NS = 2) versus one (PO, NS = 1):

- The L-BFGS approximation $H_k$ is $\gamma I$ in a 2-dimensional subspace
  versus a 1-dimensional subspace.
- The optimizer has more freedom to propose steps that move along the
  weakly-constrained gauge directions.
- If such a step moves $u_1, \dots, u_N$ far enough along the gauge, the
  phase-condition coupling (which Newton would enforce exactly) is absent,
  and the integration may become ill-conditioned.

This **does not** mean $\phi$ is flat in the gauge directions — the
gradient is nonzero there.  It means the gradient signal is **weaker**
in those directions than in the others, and the quasi-Newton approximation
cannot compensate because the phase rows of $J$ (the primary mechanism for
breaking the gauge degeneracy) are invisible to L-BFGS.

---

## 6. Why $F_{N+1}$ and $F_{N+2}$ are identically zero in the code

The implementation explicitly forces the phase-condition residuals to zero
at every evaluation:

```julia
b.d = zero.(b.d)    # in StageIterCache.update!  and  IterSolCache.update!
```

This is **not** a mathematical necessity — one could instead compute
$F_{N+1}(z) = \langle u_1 - u_{\text{ref}},\; f(u_{\text{ref}}) \rangle$
and let it become nonzero as $u_1$ moves, which would give the phase
columns of $J^\top$ a nonzero contribution to $\nabla\phi$.  The current
choice to zero them out is natural for Newton methods (where the phase
rows constrain $\delta z$ directly) but has the side effect of making
those rows invisible to L-BFGS.

---

## 7. Summary

| | Newton (hookstep) | L-BFGS |
|---|---|---|
| How $J$ is used | Solves $J\delta z = F$ | Computes $\nabla\phi = J^\top F$ |
| Phase rows of $J$ | Constrain $\delta z$ | Multiply $F.d = 0$, vanish |
| Gauge gradient | — (not needed; gauge pinned by solve) | Weak signal from continuity rows only |
| Curvature in gauge directions | Full (from $J$) | Approximated from gradient history (weak) |

The asymmetry is structural: Newton methods use the full matrix $J$ and
the phase rows actively constrain the step.  L-BFGS only sees $J^\top F$,
where the phase rows are inactive because $F.d = 0$.  The continuity rows
provide a weak gradient signal in the gauge directions, but the
curvature information L-BFGS can accumulate there is limited, especially
when two gauge directions are present.


