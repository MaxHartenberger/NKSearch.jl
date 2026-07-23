# Rebuttal: The Gauge-Drift "Issue" Is a Mathematical Error

## 1. The claim under scrutiny

The document `gauge-drift-issue.md` considers the multiple-shooting system for a
relative periodic orbit with $N$ segments, period $T$, and spatial shift $s$:

$$z = (u_1, \dots, u_N,\, T,\, s) \in \mathbb{R}^{nN+2},$$

$$F(z) = \begin{pmatrix}
\varphi(T/N, u_1) - u_2 \\
\vdots \\
\varphi(T/N, u_{N-1}) - u_N \\
S\bigl(\varphi(T/N, u_N),\, s\bigr) - u_1 \\
0 \\
0
\end{pmatrix},$$

with an augmented Jacobian $J(z) = DF(z)$ whose last two rows are the phase-condition
gradients:

$$J_{N+1,*} = (\,f(u_{\text{ref}})^\top,\; 0, \dots, 0,\; 0,\; 0\,),$$
$$J_{N+2,*} = (\,\partial_s S(u_{\text{ref}}, 0)^\top,\; 0, \dots, 0,\; 0,\; 0\,).$$

The L-BFGS method minimizes $\phi(z) = \frac{1}{2}\|F(z)\|^2$ using the gradient

$$\nabla\phi(z) = J(z)^\top F(z).$$

The central claim of the issue document is:

> *"The objective $\phi(z)$ is **exactly flat** in the two gauge directions
> spanned by*
> $$v_{\text{time}} = (f(u_1),\, 0, \dots, 0,\, 0,\, 0), \qquad
>   v_{\text{space}} = (\partial_s S(u_1, 0),\, 0, \dots, 0,\, 0,\, 0).$$

> *"Since $F_{N+1}(z) = F_{N+2}(z) = 0$ by construction, the phase columns
> contribute nothing. [...] The gauge constraints exist in $J$ but are
> invisible to $\nabla\phi$, and therefore invisible to L-BFGS."*

---

## 2. The mathematical error

The argument decomposes the gradient as

$$\nabla\phi(z) = \sum_{i=1}^{N} [J^\top]_{*,i}\; F_i(z)
                  \;+\; [J^\top]_{*,N+1}\; F_{N+1}(z)
                  \;+\; [J^\top]_{*,N+2}\; F_{N+2}(z),$$

correctly observes that $F_{N+1}(z) \equiv 0$ and $F_{N+2}(z) \equiv 0$ (by the
literal definition in the equation above), and concludes that the phase columns
of $J^\top$ contribute nothing to $\nabla\phi$.

**This is true but irrelevant.** The error is the implicit assumption that
*only* the phase columns of $J^\top$ can produce gauge-direction components of
the gradient. The **continuity columns** of $J^\top$ — columns $1$ through $N$ —
also produce nonzero gauge-direction components.

---

## 3. Direct computation of the gauge-directional gradient

Consider a single gauge direction for clarity (the time gauge; the space gauge
is analogous):

$$v = (f(u_1),\; 0, \dots, 0,\; 0,\; 0) \in \mathbb{R}^{nN+2}.$$

The component of $\nabla\phi$ along $v$ is

$$\langle \nabla\phi(z),\, v \rangle
   = \langle J(z)^\top F(z),\, v \rangle
   = \langle F(z),\, J(z)\,v \rangle.$$

Compute $J(z)\,v$ using the block-bidiagonal structure of $J$:

$$J(z)\,v = \begin{pmatrix}
\Phi_1 \cdot f(u_1) \\[2pt]
0 \\[2pt]
\vdots \\[2pt]
0 \\[2pt]
-f(u_1) \\[2pt]
\langle f(u_{\text{ref}}),\, f(u_1) \rangle \\[2pt]
\langle \partial_s S(u_{\text{ref}}, 0),\, f(u_1) \rangle
\end{pmatrix}.$$

The nonzero rows are row $1$ (continuity between segments 1 and 2),
row $N$ (continuity between segments $N$ and 1), and rows $N+1, N+2$ (phase).

Therefore

$$\boxed{
\langle \nabla\phi(z),\, v \rangle
= \langle F_1(z),\, \Phi_1 f(u_1) \rangle
+ \langle F_N(z),\, -f(u_1) \rangle
+ F_{N+1}(z)\,\langle f(u_{\text{ref}}), f(u_1) \rangle
+ F_{N+2}(z)\,\langle \partial_s S(u_{\text{ref}}, 0), f(u_1) \rangle.
}$$

**Crucially**, the continuity residuals are

$$F_1(z) = \varphi(T/N, u_1) - u_2, \qquad
  F_N(z) = \varphi(T/N, u_N) - u_1,$$

which are **nonzero** during optimization — they are precisely what the
optimizer is trying to drive to zero. Hence $\langle \nabla\phi(z), v \rangle$
is generally **nonzero**. The gradient *does* have a component in the gauge
direction, sourced from the continuity rows of $J$, not from the phase rows.

---

## 4. Why the "flatness" claim fails even at a solution

Even if one considers the behavior *at* a converged solution $z^*$ where
$F(z^*) = 0$, the objective is not flat. The curvature along $v$ is

$$\left.\frac{d^2}{d\varepsilon^2}\phi(z^* + \varepsilon v)\right|_{\varepsilon=0}
   = \|J(z^*) v\|^2.$$

Since $J(z^*) v \neq 0$ (it has nonzero entries in at least rows $1$ and $N$,
as shown above), the curvature is **strictly positive**. The point $z^*$ is a
proper isolated minimum along $v$, not a flat direction.

---

## 5. A second, subtler error: misidentification of the gauge direction

The true gauge degeneracy of the **unaugmented** multiple-shooting system
$\tilde{F}(z) = 0$ (before appending phase conditions) comes from shifting
**all** shooting points along the orbit by a common time offset:

$$u_i \;\mapsto\; \varphi(\varepsilon,\, u_i) = u_i + \varepsilon\,f(u_i) + O(\varepsilon^2), \qquad i = 1,\dots,N.$$

The tangent vector to this degeneracy is

$$w = (f(u_1),\; f(u_2),\; \dots,\; f(u_N),\; 0,\; 0),$$

**not** $v = (f(u_1), 0, \dots, 0, 0, 0)$. The vector $v$ modifies only
$u_1$ while leaving $u_2, \dots, u_N$ unchanged. This does **not** correspond to
any symmetry of the dynamics — it simply breaks the continuity conditions,
producing a large residual. The cost function $\phi$ is certainly not flat along
$v$ in any meaningful sense.

For the correct gauge tangent $w$, a similar computation shows $J(z^*) w \neq 0$
because the phase row contributes $\langle f(u_{\text{ref}}), f(u_1^*) \rangle$,
which is generically nonzero. The phase condition does its job: it breaks the
gauge degeneracy and creates a proper isolated minimum.

---

## 6. The role of the phase-condition residual

A subtle but important point: the augmented residual written as

$$F(z) = (\dots,\; 0,\; 0)$$

with literal zeros in the last two components is **notationally imprecise**.
If $F_{N+1}(z)$ were the constant-zero function, its derivative would be the
zero row vector, not $f(u_{\text{ref}})^\top$, making the Jacobian internally
inconsistent.

The mathematically correct augmented residual is

$$F_{N+1}(z) = \langle u_1 - u_{\text{ref}},\; f(u_{\text{ref}}) \rangle,$$
$$F_{N+2}(z) = \langle u_1 - u_{\text{ref}},\; \partial_s S(u_{\text{ref}}, 0) \rangle.$$

These evaluate to $0$ *at the initial guess* (by construction, since
$u_1^{(0)} = u_{\text{ref}}$), but become **nonzero** as $u_1$ moves during
optimization. In this correct formulation, the phase-condition residuals
actively contribute restoring gradient components. The $"0"$ in the
equation should be understood as shorthand for "evaluates to zero at the
starting point," not as the identically-zero function.

---

## 7. Summary of the logical error

The original argument follows this structure:

1. Phase columns of $J^\top$ multiply $F_{N+1} = F_{N+2} \equiv 0$.
2. Therefore phase columns contribute nothing to $\nabla\phi$.
3. Therefore $\nabla\phi$ has no component in the gauge directions.
4. Therefore $\phi$ is flat in the gauge directions.
5. Therefore L-BFGS cannot constrain gauge drift.

**Step 3 does not follow from step 2.** The continuity columns of $J^\top$
(columns $1$ through $N$) also produce nonzero components of $\nabla\phi$ in
the gauge directions, because the gauge vector $v$ has nonzero inner product
with continuity rows of $J$ (specifically rows $1$ and $N$).

The chain of reasoning breaks at step 3, and steps 4–5 are therefore
unsupported.

---

## 8. Conclusion

The "gauge drift" problem described in the original document is not a
genuine mathematical pathology of L-BFGS applied to phase-conditioned
multiple-shooting systems. The gradient $\nabla\phi = J^\top F$ **does**
carry information in the gauge directions — not from the phase-condition rows
of $J$, but from the continuity rows, which couple the gauge direction to the
nonzero continuity residuals $F_1, \dots, F_N$.

The objective $\phi$ is not flat in any gauge direction. The phase condition
achieves its intended purpose: it selects a unique point on the orbit by making
the augmented Jacobian full-rank and creating a proper isolated minimum of
$\phi$.
