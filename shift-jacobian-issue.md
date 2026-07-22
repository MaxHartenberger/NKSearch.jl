# Shift Operator: Transformation vs. Jacobian

## Summary

The code uses the spatial shift operator $S$ as a **transformation** in
Jacobian-vector products — it calls `S(v, s)` on a tangent perturbation
`v`, treating it as if it were a full state to be shifted.  The chain rule
actually requires the **Jacobian** $\partial_u S$ acting on `v`.

**For the OKF** the code is correct, because the OKF shift is implemented
in Fourier space as a diagonal phase multiplication $S(u,s) = e^{i k s} u$,
which is **linear** in $u$, so $\partial_u S = S(\cdot, s)$.

**For affine shifts** ($S(u,s) = u + s\mathbf{e}$, i.e. physical-space
translations) the two differ and the code would be wrong.  This limitation
is not currently exposed because (a) the OKF uses a Fourier shift and
(b) the adjoint tests that cover NS = 2 use rotation shifts.

---

## The maths (what should happen)

The closing condition on the last shooting segment is

$$
F_N(z) = S\bigl(\varphi(T/N, u_N),\, s\bigr) - u_1 .
$$

Differentiating with respect to $u_N$ gives

$$
\frac{\partial F_N}{\partial u_N}
   = \underbrace{\partial_u S(\varphi_N, s)}_{n \times n \text{ matrix}} \cdot\; \Phi_N .
$$

The object $\partial_u S$ is the **Jacobian** (derivative) of $S$ with
respect to its state argument — an $n \times n$ matrix.  Similarly,

$$
\frac{\partial F_N}{\partial T}
   = \partial_u S(\varphi_N, s) \cdot \frac{f(\varphi_N)}{N} .
$$

---

## What the code does (`IterSolCache.mul!` / `StageIterCache.mul!`)

Pseudocode for the last segment ($i = N$):

```
v  =  Φ_N · δu_N             ← tangent propagation through the flow
S(v, s)                      ← apply the shift *transformation* to v
v += δu_1                    ← wraparound identity block
v -= S(f(φ_N), s)/N · δT     ← period column
v += ∂_s S · δs              ← shift-parameter column
```

Line 2 computes $S(\Phi_N \cdot \delta u_N,\; s)$ — the perturbation is
fed through `S` as though it were a state being transformed.  The chain
rule requires $\partial_u S \cdot (\Phi_N \cdot \delta u_N)$ instead.

---

## When are they the same?

| Shift type | $S(u,s)$ | $\partial_u S$ | $S(v,s)$ vs $\partial_u S \cdot v$ |
|---|---|---|---|
| **Fourier multiplier** (OKF) | $e^{i k s}\,u$ | $e^{i k s}$ | **identical** |
| Rotation matrix | $R(s)\,u$ | $R(s)$ | **identical** |
| Physical translation | $u + s\mathbf{e}$ | $I$ | $S(v,s) = v + s\mathbf{e} \;\neq\; v = I\cdot v$ |

A Fourier multiplier is a **linear** operator (diagonal in spectral space).
For any linear operator, $\partial_u S = S(\cdot, s)$.  The code is correct
whenever $S$ is linear in its state argument.

A physical-space translation is **affine** (linear part $I$, constant part
$s\mathbf{e}$).  For an affine shift the code would add the spurious
constant $s\mathbf{e}$ to the perturbation — an artefact of treating the
perturbation as a full state.

The criterion is **linearity in $u$**, not $S(0,s)=0$ (the latter follows
from linearity but does not imply it).

---

## The adjoint (transpose)

In `AdjointIterSolCache.mul!` the adjoint applies

```
S(w[N], -s)                  ← inverse transformation
```

*before* the backward adjoint integration, to implement $(\partial_u S)^\top$.
For the OKF's Fourier shift this is correct:

$$
(\partial_u S)^\top = \bigl(e^{i k s}\bigr)^* = e^{-i k s} = S(\cdot, -s)
$$

For an affine translation it would be wrong:

$$
(\partial_u S)^\top = I^\top = I \;\neq\; S(\cdot,-s) = v \mapsto v - s\mathbf{e}.
$$

---

## Status for the current codebase

### OKF — correct ✓

The OKF shift `xshift!` multiplies each Fourier coefficient by
$e^{i k_y s}$, which is a diagonal linear operator.  Therefore
$\partial_u S = S(\cdot, s)$ and $(\partial_u S)^\top = S(\cdot, -s)$
hold exactly.  The gradient FD check (`test_lbfgs_RPO.jl`) confirms this
with $\text{rel\_err} \sim 10^{-9}$.

### Test coverage gap

- `test_adjoint.jl` §6 tests the NS = 2 adjoint identity with a **rotation**
  shift (`SpatialShift`), which is also linear — the test passes.
- `test_search_shift.jl` includes a physical **translation** (`ZShift`) for
  the drift-Hopf system, but the L‑BFGS convergence check is commented out
  and the hookstep test is wrapped in `try`/`catch`.
- No test validates the Jacobian or gradient for an affine shift.

### Report (main_okf.tex)

Section 3.4 states that $\partial_u S = S(\cdot, s)$ because
"$S$ is a homogeneous linear map: $S(0,s)=0$."  This is **correct in
conclusion but imprecise in reasoning**:

- $S(0,s)=0$ is a consequence of linearity, not a sufficient condition.
- The correct justification is: for the OKF, $S$ is a diagonal Fourier
  multiplier and therefore **linear** in $u$, so its Jacobian is itself.
- The report should also note that this identity is specific to linear
  shifts and does not generalise to arbitrary shift operators.

---

## What a general fix would require (if affine shifts are ever needed)

The shift API currently provides two callbacks:

| Callback | Signature | Purpose |
|---|---|---|
| `S` | `S(x, s)` — transform state `x` by `s` | residual evaluation |
| `dS` | `dS(out, x)` — $\partial_s S(x,0)$ | phase condition |

Supporting affine shifts would need a **third** callback:

| Callback | Signature | Purpose |
|---|---|---|
| `dS_du` | `dS_du(v, x, s)` — apply $\partial_u S(x,s)$ to `v` | Jacobian-vector products |

Then every `S(v, s)` in the Jacobian paths would become `dS_du(v, φ_N, s)`,
and every `S(w, -s)` in the adjoint would become `dS_du^T(w, φ_N, s)`.

For linear shifts (OKF, rotations): `dS_du(v, x, s) = S(v, s)` — backward
compatible.  For affine translations: `dS_du(v, x, s) = v` (the identity).

This is **not needed** for the current OKF workflow.
