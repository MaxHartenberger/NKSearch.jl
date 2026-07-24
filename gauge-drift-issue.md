# Gauge Drift in L-BFGS for Relative Periodic Orbits

## 1. Multiple-shooting formulation

A relative periodic orbit (RPO) of period $T$ and spatial shift $s$ is a
root of the augmented multiple-shooting system

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
reference state $u_{\text{ref}} = u_1^{(0)}$ (the initial $u_1$):

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
and $\widehat f_N = \partial_u S(\varphi_N, s)\,f(\varphi_N)$.  For a Fourier shift
$S(u,s) = e^{i k s}u$, we have $\partial_u S = S(\cdot, s)$.

---

## 2. The gauge degeneracy

The **unaugmented** system (rows $1$ through $N$ only) has a continuous
degeneracy: shifting **all** shooting points by a common time offset $\tau$
leaves the continuity equations satisfied to first order.  The tangent to
this family is

$$
w_{\text{time}} = \bigl(f(u_1),\; f(u_2),\; \dots,\; f(u_N),\; 0,\; 0\bigr).
$$

For a near-RPO the linearised flow approximately maps tangent vectors:
$\Phi_i\,f(u_i) \approx f(u_{i+1})$, so

$$
(Jw_{\text{time}})_i = \Phi_i f(u_i) - f(u_{i+1}) \approx 0 \qquad (i = 1,\dots,N-1).
$$

The closing segment contributes $(Jw_{\text{time}})_N = \widehat\Phi_N f(u_N) - f(u_1)$,
also small for a near-RPO.  Analogously a common spatial shift generates

$$
w_{\text{space}} = \bigl(\partial_s S(u_1,0),\; \dots,\; \partial_s S(u_N,0),\; 0,\; 0\bigr).
$$

The directions $w_{\text{time}}$ and $w_{\text{space}}$ are **approximate**
null directions of the continuity rows of $J$ — they become exact null
directions in the limit of a converged RPO.

---

## 3. How Newton methods pin the gauge

The hookstep / trust-region Newton method solves

$$
J(z_k)\,\delta z = F(z_k)
$$

at each iteration.  The **phase rows of $J$ actively constrain the Newton
step** $\delta z$.  The row $f(u_{\text{ref}})^\top$ forces

$$
\langle \delta u_1,\; f(u_{\text{ref}}) \rangle = 0,
$$

which, together with the coupling through the continuity rows, prevents
the step from moving along the gauge direction $w_{\text{time}}$.  The
gauge is pinned by the matrix structure of $J$.

---

## 4. How L-BFGS sees the Jacobian

L-BFGS minimises $\phi(z) = \frac{1}{2}\|F(z)\|^2$ using only gradient
information:

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

In the current implementation, $F_{N+1}$ and $F_{N+2}$ are **explicitly
set to zero** at every residual evaluation (`b.d = zero.(b.d)`).  Hence
the phase columns of $J^\top$ contribute **nothing** to $\nabla\phi$ at
any iteration.

### 4.2 Continuity columns do contribute — but weakly

The **continuity columns** of $J^\top$ (columns $1$ through $N$) multiply
the continuity residuals $F_1,\dots,F_N$, which are nonzero during
optimization.  Computing the component of $\nabla\phi$ along $w_{\text{time}}$:

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
from the continuity rows is therefore **weak**.

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
  nonlinear integration may become ill-conditioned, and the line search
  may evaluate trial points that diverge.

---

## 6. Why $F_{N+1}$ and $F_{N+2}$ are forced to zero

The implementation explicitly sets the phase-condition residuals to zero
at every evaluation:

```julia
b.d = zero.(b.d)    # in StageIterCache.update!  and  IterSolCache.update!
```

This is natural for Newton methods, where the phase rows of $J$ constrain
$\delta z$ directly through the linear solve.  The RHS value of those rows
has negligible effect — what matters is that they are present in $J$.

For L-BFGS, the phase rows are invisible to $\nabla\phi$ because they
multiply $F.d = (0,0)$.  The gauge constraints exist in $J$ but are not
used in a way that helps L-BFGS.

---

## 7. Summary of the asymmetry

| | Newton (hookstep) | L-BFGS |
|---|---|---|
| How $J$ is used | Solves $J\delta z = F$ | Computes $\nabla\phi = J^\top F$ |
| Phase rows of $J$ | Constrain $\delta z$ | Multiply $F.d = 0$, vanish |
| Gauge-direction gradient | — (not needed; gauge pinned by solve) | Weak signal from continuity rows only |
| Curvature in gauge directions | Full (from $J$) | Approximated from gradient history (weak) |

The asymmetry is structural: Newton methods use the full matrix $J$ and
the phase rows actively constrain the step.  L-BFGS only sees $J^\top F$,
where the phase rows are inactive because $F.d = 0$.  The continuity rows
provide a weak gradient signal in the gauge directions, but the curvature
information L-BFGS can accumulate there is limited, especially with two
gauge directions.

---

## 8. Remedy: compute actual phase-condition residuals

Instead of zeroing the phase-condition residuals, evaluate them as proper
functions of $z$:

**Before:**
```julia
b.d = zero.(b.d)   # F_{N+1}, F_{N+2} := 0
```

**After:**
```julia
# In update!(cache, b, z):
uref = cache.phase_ref                    # frozen at construction
b.d[1] = dot(z[1] - uref, f(uref))       # time-phase residual
b.d[2] = dot(z[1] - uref, ∂ₛS(uref, 0))  # space-phase residual (NS == 2)
```

### Mathematical effect

The gradient along $w_{\text{time}}$ gains an additional term:

$$
\langle \nabla\phi, w_{\text{time}} \rangle
   = \sum_{i=1}^{N} \langle F_i,\; \Phi_i f(u_i) - f(u_{i+1}) \rangle
   \;+\; F_{N+1}(z)\,\langle f(u_{\text{ref}}), f(u_1) \rangle,
$$

where $F_{N+1}(z) = \langle u_1 - u_{\text{ref}},\; f(u_{\text{ref}}) \rangle$.

Crucially, $F_{N+1}(z)$ is $O(\|z - z^*\|)$ — **first-order** in the
distance from the solution, matching the scaling of non-gauge gradient
components.  The factor $\langle f(u_{\text{ref}}), f(u_1) \rangle$ does
not vanish at the solution (it approaches $\|f(u^*_1)\|^2 \neq 0$).

### What to store

A `phase_ref` field is added to the solver cache structs, set to
`deepcopy(z0[1])` at construction time.  The frozen reference quantities
$f(u_{\text{ref}})$ and $\partial_s S(u_{\text{ref}}, 0)$ are computed
on the fly using the existing derivative operators $D[1]$ and $D[2]$
and `phase_ref`.

### Safety for Newton methods

For Newton (hookstep, trust-region) methods, the phase rows of $J$
already constrain $\delta z$ through the linear solve.  Changing
$F_{N+1}$ from $0$ to a small nonzero value only changes the RHS —
the *direction* of the constraint is unaffected.  In practice, the
fix should initially be applied only to L-BFGS (`StageIterCache`),
not to Newton caches (`IterSolCache`, `DirectSolCache`), where it is
unnecessary and may interact poorly with trust-region logic.

---

## 9. Implementation notes

### Files to modify

| File | Change |
|---|---|
| `src/lbfgs_sol_cache.jl` | Add `phase_ref` to `StageIterCache` and `AdjointIterSolCache`; compute actual phase residuals in `update!`; use `phase_ref` in forward/adjoint `mul!` phase rows |
| `src/iter_sol_cache.jl` | Add `phase_ref` field (unused — keep `b.d = zero.(b.d)`) |
| `src/direct_sol_cache.jl` | Same |
| `src/newton.jl` | Pass `fwd_cache.phase_ref` to `AdjointIterSolCache` constructor |

### Backward compatibility

For Newton and hookstep methods the behavior is unchanged.  For L-BFGS
the gradient becomes mathematically complete — every row of $J$
contributes to $\nabla\phi$ through the corresponding residual component
it differentiates, restoring the internal consistency of the augmented
system:

$$F_{N+1}(z) = \langle u_1 - u_{\text{ref}},\; f(u_{\text{ref}}) \rangle, \qquad
\frac{\partial F_{N+1}}{\partial u_1}(z) = f(u_{\text{ref}})^\top.$$

### Known issues

- **Memory**: `deepcopy(z0[1])` duplicates FFTW plans.  For the OKF
  ($100 \times 100$ grid) the extra allocation is negligible (~80 KB per
  segment), but the gradient-check test code that deepcopies all flow
  operators must be guarded with an `--enable-gradient-check` flag to
  avoid OOM in production runs.
- **Newton interference**: applying the fix to Newton caches can cause
  the trust-region radius to shrink prematurely when the phase residual
  conflicts with the continuity equations.  Keep the fix L-BFGS–only.
