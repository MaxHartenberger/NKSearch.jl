# L-BFGS RPO Convergence Failure — Diagnostic Report
#
# Prepared for NKSearch.jl developers.
# Based on a run of 89 perturbed OKF relative periodic orbits (RPOs)
# through L-BFGS optimisation (method=:lbfgs_opt).
#
# Environment: Julia 1.12.4, NKSearch (master), Flows, OpenKolmogorovFlow
# OKF params:  Re=40, n=33 Fourier cutoff, m=49 dealiased grid, Δt=0.01
# Threads: 4, Memory: 8 GB (OOM-killed; average RSS ~2 GB before spike)
# ============================================================================

SUMMARY
=======

The L-BFGS optimiser fails on ALL perturbed RPO initial conditions.
Three distinct failure modes were observed, each triggered at a different
stage.  The first failure corrupts shared multi-threading state (Channels
in StageIterCache), causing a cascade that makes all subsequent ICs fail
immediately.  The job was ultimately OOM-killed, likely due to NaN
propagation into unbounded allocations.

================================================================================
FAILURE MODE 1 — L-BFGS Divergence → NaN (IC 1, N=1)
================================================================================

IC 1 (tmpHafpc5_perturbed, N=1 segment, T≈24.41) ran for 29 iterations
before hitting NaN.  The iteration table shows:

  iter   ||∇ϕ||      ||F||
  --------------------------
    0    2.44e-01    9.76e-02    ← starting well
    8    1.19e-01    1.27e-02
    9    2.23e-02    7.03e-03
   10    1.80e-02    6.25e-03
   11    2.13e-02    5.78e-03
   12    2.23e-02    4.51e-03
   13    3.40e-02    4.29e-03
   14    1.37e-02    3.39e-03
   15    3.62e-03    3.03e-03    ← best point (||F|| ~ 3e-3)
   16    2.98e-03    2.99e-03
   17    7.03e-03    3.04e-03    ← slight uptick
   18    2.05e-01    1.71e-02    ← SUDDEN JUMP (10× increase in ||F||)
   19    1.38e+00    9.90e-02    ← exploded
   20    3.06e-01    2.55e-02
   ...
   28    9.22e+00    1.85e+00    ← diverging
   29       NaN        NaN       ← dead

Key observations:
  - The L-BFGS was converging steadily (||F|| dropped from 9.8e-2 to 3.0e-3
    over 16 iterations).
  - At iteration 18, the gradient and residual suddenly jumped by 10–100×.
    This suggests the line search took a step into a region where the
    nonlinear flow integration became inaccurate or unstable.
  - Once the residual jumped, the L-BFGS history (s,y pairs) became
    contaminated with bad curvature information, and the method could not
    recover.
  - At iteration 29, NaN appeared, likely from the nonlinear flow G
    producing NaN in the state, which then propagated through the adjoint
    L_adj (J^T · w involves the state-dependent Jacobian).

The NaN then triggered:

  ERROR: CompositeException(Any[TaskFailedException(Task (failed) @0x...)])

The crash site (from stderr):

  sync_end(c::Channel{Any}) at task.jl:630
  update!(mm::StageIterCache{..., 1, 2, ...})      ← segment 1 of 2?
  compute_gradient! at search_lbfgs.jl:67
  _search_lbfgs_opt!(Gs, Ls, S, D, z0, ...)
  _search!(Gs, Ls, Ls_adj, S, D, z0, ...)
  search!(G, L, L_adj, S, F_phase, ddx!, z, opts)

The crash happens inside update!() on a StageIterCache.  The
StageIterCache uses Julia Channels for multi-threaded synchronisation
when populating the stage caches from the nonlinear flow G.  When G
produces NaN, a spawned task fails, and the Channel's sync_end() throws
a TaskFailedException that propagates up as a CompositeException.

================================================================================
FAILURE MODE 2 — Immediate TaskFailedException (ICs 2 & 3, N=2, N=3)
================================================================================

After IC 1 failed, ICs 2 and 3 crashed immediately at iteration 0:

  [ 2/89] tmpjNHhe6_perturbed  N=2  T_csv=22.72
    |   0  |  lbfgs | 7.711e+01 | 3.030e-01 |
    ERROR: CompositeException(Any[TaskFailedException(Task (failed) @0x...),
                                   TaskFailedException(Task (failed) @0x...)])

  [ 3/89] tmptqXIrh_perturbed  N=3  T_csv=56.47
    |   0  |  lbfgs | 4.204e+12 | 2.957e-01 |   ← gradient = 4×10¹² !!

Key observations:
  - IC 2 failed BEFORE any L-BFGS iteration could complete (only iter 0
    printed).  The two TaskFailedExceptions suggest BOTH shooting
    segments (N=2) failed simultaneously.
  - IC 3 computed iteration 0 but with a gradient norm of 4×10¹² —
    clearly garbage, indicating the adjoint integration produced nonsense
    before crashing.
  - The flow operators G, L, L_adj are created ONCE in
    setup_dynamics_lbfgs() and reused for all 89 ICs.  After IC 1's NaN
    crash, the StageIterCache's internal Channel state was left
    corrupted.  This corrupted state poisoned all subsequent ICs.

================================================================================
FAILURE MODE 3 — OOM Kill
================================================================================

The SLURM epilogue reports:

  State: OUT_OF_MEMORY (exit code 0)
  Memory Utilized: 2.00 GB
  Memory Efficiency: 24.96% of 8.00 GB

The 2 GB is the average RSS over the 12.5-minute job lifetime.  The OOM
kill was a sudden spike past 8 GB, likely caused by:
  - NaN propagating into array allocations (e.g., the L-BFGS direction
    vector or s,y history matrices being filled with NaN, then used in
    BLAS operations that allocate workspace).
  - Corrupted Channel state causing resource leaks (unreaped tasks
    holding onto memory).
  - The CompositeException unwinding may have left FFTW plans or stage
    cache arrays in an inconsistent state.

================================================================================
ROOT CAUSE ANALYSIS
================================================================================

1.  No NaN guarding in _search_lbfgs_opt!
    ---------------------------------------
    The L-BFGS loop does not check for NaN or Inf in the residual ||F||,
    the gradient ||∇ϕ||, or the state vector z.  When the nonlinear flow
    G produces NaN (e.g., from a large step taken by the line search),
    the NaN silently enters the L-BFGS history (s,y pairs), the gradient
    computation, and the direction vector.  The line search then tries to
    evaluate φ(z + λ·d) with NaN direction, producing more NaN, and the
    method spirals.

    Recommended fix: After each call to compute_gradient!() and after
    each line-search trial, check isnan(ϕ) and isnan(norm(gradient)).
    If NaN is detected, abort the search gracefully and return an error
    status (e.g., :nan_detected).

2.  State corruption in StageIterCache across search! calls
    --------------------------------------------------------
    The StageIterCache uses Julia Channels (@spawn/@sync) for
    multi-threaded population of RK4 stage caches.  When a spawned task
    fails (e.g., because G produced NaN and a downstream operation
    crashed), the Channel is left in a broken state.  The next call to
    update!() on the SAME StageIterCache instance hangs or throws
    TaskFailedException because the Channel was never reset.

    The flow operators are created once and reused:

        G, L, L_adj, S, F_phase, F_raw = setup_dynamics_lbfgs()
        for each IC ...
            search!(G, L, L_adj, S, F_phase, ddx!, z, opts)
        end

    After the first search!() fails, the StageIterCache inside G (and
    possibly L, L_adj) is corrupted.  Subsequent search!() calls fail
    immediately.

    Recommended fixes:
    a) In _search_lbfgs_opt!, wrap the main loop in a try/finally that
       resets the StageIterCache state (re-initialise Channels, clear
       stage arrays) on error.
    b) Alternatively, provide a `reset!(flow)` or `reset!(cache)` API
       that callers can invoke between ICs.
    c) As a workaround, callers can reconstruct the flow operators for
       each IC, but this is expensive (FFTW plan creation).

3.  No convergence status returned
    -------------------------------
    _search_lbfgs_opt! always returns `nothing`, unlike Newton methods
    which return symbols like :converged, :maxiter_reached, etc.
    Callers must manually re-compute ||F(z)|| after search! returns and
    compare against e_norm_tol.  This is manageable but error-prone.

    Recommended fix: Return a status symbol consistent with the Newton
    methods.  Add :nan_detected, :diverged, and :task_failed statuses.

4.  NS==2 (RPO spatial shift) explicitly blocked
    ---------------------------------------------
    Per NKSearch commit 0e6579c (2026-06-19), AdjointIterSolCache.mul!
    throws an intentional error for NS==2:

      "AdjointIterSolCache: spatial-shift transpose (NS == 2)
       is not yet implemented."

    This means L-BFGS on RPOs with N≥2 segments is impossible until
    this is implemented.  The N=1 case worked (for 29 iterations)
    because with a single segment there are no inter-segment spatial
    shift couplings in the Jacobian transpose.

    The code to compute the spatial-shift adjoint contribution already
    exists in the same function (out_d_2 is computed).  Only the final
    tuple assembly and the error() guard need to be changed.

5.  compute_residual discrepancy
    -----------------------------
    The user's compute_residual() (in test_lbfgs.jl) reported
    residual_init = 3.02×10⁹ for IC 1, while the L-BFGS iteration table
    showed ||F|| = 9.76×10⁻² at iteration 0 — a difference of 10
    orders of magnitude.  This suggests either:
    a) The user's compute_residual() and the internal ||F|| in NKSearch
       use different normalisations or segment-count scaling, or
    b) compute_residual() has a bug (e.g., not dividing the segment
       duration correctly).

================================================================================
REPRODUCTION
================================================================================

To reproduce, run the attached test script against any perturbed IC:

  julia -t 4 scripts/test_lbfgs.jl --max-test 3 --maxiter 100

Expected: IC 1 runs for ~30 iterations then hits NaN/TaskFailedException.
ICs 2–3 crash immediately at iteration 0.

The perturbed IC files and manifest are in:
  initial_conditions/perturbed/
  initial_conditions/perturbed/perturbation_manifest.csv

================================================================================
STDERR EXCERPTS (de-truncated)
================================================================================

Stack trace 1 (segment 1 crash):
  sync_end(c::Channel{Any}) at task.jl:630
  macro expansion at task.jl:663 [inlined]
  update!(mm::StageIterCache{FTField{33,49,...}, 1, 2, ...})
     ← segment 1 of 2
  compute_gradient! at search_lbfgs.jl:67 [inlined]
  _search_lbfgs_opt!(Gs, Ls, S, D, z0, ...)
  _search!(Gs, Ls, Ls_adj, S, D, z0, ...)
  search!(G, L, L_adj, S, F_phase, ddx!, z, opts)

Stack trace 2 (segment 2 crash):
  sync_end(c::Channel{Any}) at task.jl:630
  macro expansion at task.jl:663 [inlined]
  update!(mm::StageIterCache{FTField{33,49,...}, 2, 2, ...})
     ← segment 2 of 2
  compute_gradient! at search_lbfgs.jl:67 [inlined]

Both crashes originate at search_lbfgs.jl line 67 in compute_gradient!,
which calls update!() on the StageIterCache to populate the nonlinear
stage caches before the adjoint pass.

================================================================================
PRIORITY ORDER FOR FIXES
================================================================================

  [CRITICAL]  Add NaN/Inf guards in _search_lbfgs_opt!
  [CRITICAL]  Reset StageIterCache Channel state on error (or provide
              reset! API)
  [HIGH]      Return convergence status from _search_lbfgs_opt!
  [HIGH]      Implement NS==2 spatial-shift transpose in
              AdjointIterSolCache.mul!
  [MEDIUM]    Add defensive isnan() checks in compute_gradient! before
              spawning tasks
