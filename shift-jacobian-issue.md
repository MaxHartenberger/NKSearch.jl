# Shift Operator: Transformation vs. Jacobian

## Summary

The code (and Section 4 of the report) uses the spatial shift operator $S$
as a **transformation** in Jacobian-vector products, but the chain rule
requires the **Jacobian** $\partial_u S$.  These coincide only when $S$ is
a homogeneous linear operator ($S(0,s)=0$), e.g. rotations.  For affine
shifts like translations the Jacobian is silently wrong.

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
respect to its state argument.  It is an $n \times n$ matrix that answers:
*if I perturb the state by $\delta u$, how does the shifted output change?*

Similarly, the time-derivative column involves

$$
\frac{\partial F_N}{\partial T}
   = \partial_u S(\varphi_N, s) \cdot \frac{f(\varphi_N)}{N} .
$$

---

## What the code does

In `IterSolCache.mul!` and `StageIterCache.mul!`, the last segment is
handled by (pseudocode):

```
v = Φ_N · δu_N            ← tangent propagation through the flow
S(v, s)                   ← apply the shift *transformation* to v
v += δu_1                 ← wraparound identity block
v -= S(f(φ_N), s)/N · δT  ← period column
v += ∂_s S · δs           ← shift-parameter column
```

Line 2 computes $S(\Phi_N \cdot \delta u_N,\; s)$ — it feeds the
perturbation through the shift as if it were a **state** being
transformed.  But the chain rule says it should be
$\partial_u S \cdot (\Phi_N \cdot \delta u_N)$ — multiplication by the
**Jacobian matrix**.

---

## When are they the same?

| Shift type | $S(u,s)$ | $\partial_u S$ | $S(v,s)$ vs $\partial_u S \cdot v$ |
|---|---|---|---|
| Rotation | $R(s)\,u$ | $R(s)$ | identical |
| Translation | $u + s\mathbf{e}$ | $I$ | $S(v,s) = v + s\mathbf{e} \;\neq\; v = I\cdot v$ |

For a **translation**, the code adds a spurious constant vector $s\mathbf{e}$
to the propagated perturbation.  This constant does not belong in a
Jacobian — it is an artefact of treating the perturbation as though it
were a full state to be translated.

---

## The adjoint (transpose) inherits the same issue

In `AdjointIterSolCache.mul!` the adjoint applies

```
S(w[N], -s)               ← inverse transformation
```

*before* the backward adjoint integration, intending to implement
$(\partial_u S)^\top$.  For a translation:

$$
(\partial_u S)^\top = I^\top = I \;\neq\; S(\cdot,-s) = v \mapsto v - s\mathbf{e}.
$$

---

## Where the report is inconsistent

- **Section 2 (Problem Formulation), Eq. (3):** correctly defines
  $\widehat\Phi_N := \partial_u S(\varphi_N,s)\,\Phi_N$ using the Jacobian.
  This is the right derivative.

- **Section 4 (Adjoint flow with spatial shift):** switches to describing
  the implementation and writes

  > For a translation, $\widehat S^\top = \widehat S^{-1} = S(\cdot,-s)$,
  > i.e. the inverse shift.  This is the key identity used in the
  > implementation.

  But $\widehat S$ was defined as $\partial_u S = I$ for a translation.
  The identity matrix equals neither $S(\cdot,s)$ (which is $v\mapsto v+s\mathbf{e}$)
  nor $S(\cdot,-s)$.  The report is silently swapping the Jacobian for the
  transformation and claiming they are the same — which is only true for
  homogeneous linear shifts.

---

## Why the tests still pass

The test problem (`test/test_search_shift.jl`) uses a **rotation**
(spatial shift = rotating each Fourier mode by $k\cdot s$).  For a
rotation $S(0,s)=0$, so the transformation and its Jacobian coincide
and the bug is masked.

A test with a **translation** ($S(u,s) = u + s\mathbf{e}$) would expose
the issue immediately.

---

## What a correct fix would require

The shift API currently provides two callbacks:

| Callback | Signature | Purpose |
|---|---|---|
| `S` | `S(x, s)` — transform state `x` by `s` | residual evaluation |
| `dS` | `dS(out, x)` — $\partial_s S(x,0)$ | phase condition |

A complete fix needs a **third** callback for the state-Jacobian:

| Callback | Signature | Purpose |
|---|---|---|
| `dS_du` | `dS_du(v, x, s)` — apply $\partial_u S(x,s)$ to `v` | Jacobian-vector products |

Then in all `mul!` methods, every occurrence of

```julia
S(v, s)          # transformation (wrong for derivatives)
```

would be replaced by

```julia
dS_du(v, φ_N, s)  # Jacobian (correct)
```

and in the adjoint, every

```julia
S(w, -s)          # inverse transformation (wrong for derivatives)
```

would be replaced by

```julia
dS_du^T(w, φ_N, s)  # transpose of Jacobian
```

For a rotation, `dS_du(v, x, s) = S(v, s)` — backward compatible.
For a translation, `dS_du(v, x, s) = v` (identity), while `S(v, s) = v + s𝐞`.

---

## What to change in the report

Option A (preferred if the limitation is accepted):
- In Section 4, add a remark that the identities $\widehat S^\top = S(\cdot,-s)$
  hold only when $S$ is a homogeneous linear operator ($S(0,s)=0$ for all $s$),
  and that this is the class of shifts the implementation supports.

Option B (if the code is generalised):
- Introduce separate notation for the transformation $S$ and its Jacobian
  $\partial_u S$ throughout, and rewrite Section 4 to derive the adjoint
  using the Jacobian rather than the transformation.
