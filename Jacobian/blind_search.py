"""
Blind marked-factor search for a polynomial Jacobian counterexample.

The search is blind with respect to the final certificate: it does not assume
a coefficient slice, boundary modulus, polynomial chart, degree-seven map, or
specific collision witness.

The structural input is the marked factorization

```
L = a*U + b*V,
Q = c*U**2 + d*U*V + e*V**2,
```

together with its visible cubic coefficients and the resultant normalization
R(L,Q) = 1, which removes the continuous scaling gauge.

STAGE 1 -- chart discovery.

```
Enumerate the four visible-coefficient slices M_i = 1 and the five
possible modulus variables. For each combination, solve the slice and
resultant constraints, analyze the modulus-zero boundary, introduce
residual coordinates, and repeatedly resolve poles through polynomial
blowups. Accept the first resulting chart whose induced three-dimensional
map has a nonzero constant Jacobian determinant.
```

STAGE 2 -- collision construction.

```
Enumerate rational cubics with three distinct roots satisfying the
discovered slice condition. Mark each of the three roots in turn as the
linear factor, normalize each marked factorization by the resultant, pull
the states back through the discovered chart, and verify that three
distinct source points have exactly the same image.
```

All algebraic checks use exact SymPy arithmetic. No coefficients or collision
points from the known counterexample are embedded in the search.
"""

import multiprocessing as mp
import itertools
import sympy as sp

a, b, c, d, e = sp.symbols('a b c d e')
U, V = sp.symbols('U V')
VARS = [a, b, c, d, e]
NAMES = {a: 'a', b: 'b', c: 'c', d: 'd', e: 'e'}
SYM = {'a': a, 'b': b, 'c': c, 'd': d, 'e': e}

L = a*U + b*V
Q = c*U**2 + d*U*V + e*V**2
R = sp.resultant(L.subs(V, 1), Q.subs(V, 1), U)

Mexprs = {
    'M1': a*c,
    'M2': a*d + b*c,
    'M3': a*e + b*d,
    'M4': b*e,
}


# ---------------------------------------------------------------------------
# STAGE 1: discover the (hyperplane, modulus) slice
# ---------------------------------------------------------------------------
def attempt_chart(hyperplane_name, modname, max_blowups=8):
    mod = SYM[modname]
    Hi = Mexprs[hyperplane_name]
    m = mod
    others = [v for v in VARS if v != m]

    for p in others:
        p_sols = sp.solve(sp.Eq(Hi, 1), p)
        if len(p_sols) != 1:
            continue
        p_expr = p_sols[0]

        remaining1 = [v for v in others if v != p]
        for q in remaining1:
            R_sub = R.subs(p, p_expr)
            q_sols = sp.solve(sp.Eq(R_sub, 1), q)
            if len(q_sols) != 1:
                continue
            q_expr = sp.simplify(q_sols[0])

            free_vars = [v for v in remaining1 if v != q]
            if len(free_vars) != 2:
                continue
            f1, f2 = free_vars

            eq1_0 = (Hi - 1).subs(m, 0)
            eq2_0 = (R - 1).subs(m, 0)
            b_sols = sp.solve([eq1_0, eq2_0], [f1, f2], dict=True)
            b_sols = [s for s in b_sols if f1 in s and f2 in s]
            if not b_sols:
                continue
            f1_0, f2_0 = b_sols[0][f1], b_sols[0][f2]

            y = sp.symbols(f'y_{NAMES[f1]}')
            ctemp = sp.symbols('ctmp')
            f1_alg = f1_0 + m*y
            f2_alg = f2_0 + m*ctemp

            subs_map = {f1: f1_alg, f2: f2_alg}
            p_alg = sp.simplify(p_expr.subs(subs_map))
            q_alg = sp.simplify(q_expr.subs(subs_map))

            current_temp = ctemp
            cleared = False
            for i in range(max_blowups):
                q_expanded = sp.expand(q_alg)
                if not q_expanded.has(1/m):
                    cleared = True
                    break
                if sp.count_ops(q_expanded) > 300:
                    break
                pole_num = sp.simplify(q_expanded * m).subs(m, 0)
                if pole_num == 0:
                    break
                new_var = sp.symbols(f'z_{NAMES[f1]}_{i}')
                sol = sp.solve(sp.Eq(pole_num, m*new_var), current_temp)
                if not sol:
                    break
                val = sol[0]
                p_alg = sp.expand(p_alg.subs(current_temp, val))
                q_alg = sp.expand(q_alg.subs(current_temp, val))
                f2_alg = sp.expand(f2_alg.subs(current_temp, val))
                current_temp = new_var

            if not cleared:
                continue

            z = current_temp
            full = {m: m, f1: f1_alg, f2: f2_alg, p: p_alg, q: q_alg}
            other_Ms = [Mexprs[k] for k in Mexprs if k != hyperplane_name]
            G = [sp.expand(expr.subs(full)) for expr in other_Ms]
            Jmat = sp.Matrix(G).jacobian([m, y, z])
            detJ = sp.simplify(Jmat.det())

            if detJ.is_constant() and detJ != 0:
                return dict(hyperplane=hyperplane_name, modulus=NAMES[m],
                            p=p, q=q, f1=f1, f2=f2, m=m, y=y, z=z,
                            full=full, G=G, detJ=detJ)
    return None


def _worker(hp, modname, out_q):
    try:
        out_q.put(attempt_chart(hp, modname))
    except Exception:
        out_q.put(None)


def discover_slice(per_task_timeout=6, verbose=True):
    for hp in Mexprs:
        for modname in NAMES.values():
            q = mp.Queue()
            proc = mp.Process(target=_worker, args=(hp, modname, q))
            proc.start()
            proc.join(per_task_timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                if verbose:
                    print(f"{hp} / modulus {modname}: timed out -- skipped")
                continue
            res = q.get() if not q.empty() else None
            if res is None:
                if verbose:
                    print(f"{hp} / modulus {modname}: no valid chart")
                continue
            if verbose:
                print(f"{hp} / modulus {modname}: HIT, constant Jacobian = "
                      f"{res['detJ']}  (p={NAMES[res['p']]}, "
                      f"q={NAMES[res['q']]})")
            return res  # return the FIRST hit found -- stage 2 runs on this
    return None


# ---------------------------------------------------------------------------
# STAGE 2: collision search on the discovered chart, via marking symmetry
# ---------------------------------------------------------------------------
def marked_states(r1, r2, r3):
    roots = [r1, r2, r3]
    out = []
    for i in range(3):
        ri = roots[i]
        rj, rk = [roots[j] for j in range(3) if j != i]
        out.append({a: sp.Integer(1), b: -ri, c: sp.Integer(1),
                    d: -(rj+rk), e: rj*rk})
    return out


def gauge_fix(state):
    a_, b_, c_, d_, e_ = state[a], state[b], state[c], state[d], state[e]
    Rv = a_**2*e_ - a_*b_*d_ + b_**2*c_
    if Rv == 0:
        return None
    lam = 1/Rv
    return {a: lam*a_, b: lam*b_, c: c_/lam, d: d_/lam, e: e_/lam}


def invert_chart(slice_res, state):
    """Solve for (free1,free2) given numeric (a,b,c,d,e) on the target locus,
    using the chart discovered in stage 1 -- no closed-form inverse assumed."""
    m, f1, f2, p, q = slice_res['m'], slice_res['f1'], slice_res['f2'], \
                      slice_res['p'], slice_res['q']
    full = slice_res['full']
    mval = state[m]
    f1_expr = full[f1].subs(m, mval)
    f2_expr = full[f2].subs(m, mval)
    sols = sp.solve([sp.Eq(f1_expr, state[f1]), sp.Eq(f2_expr, state[f2])],
                     [slice_res['y'], slice_res['z']], dict=True)
    p_expr = full[p].subs(m, mval)
    q_expr = full[q].subs(m, mval)
    for s in sols:
        if sp.simplify(p_expr.subs(s) - state[p]) == 0 and \
           sp.simplify(q_expr.subs(s) - state[q]) == 0:
            return mval, s[slice_res['y']], s[slice_res['z']]
    return None


def invariant_condition_on_roots(hyperplane_name, r1, r2, r3):
    """The hyperplane condition M_i=1, expressed on the roots -- this is
    what tells the outer loop which root-triples are worth trying."""
    e1 = r1 + r2 + r3
    e2 = r1*r2 + r1*r3 + r2*r3
    e3 = r1*r2*r3
    return {'M1': sp.Integer(1) - 1,           # M1=ac is fixed =1 by raw markings
            'M2': -e1 - 1,
            'M3': e2 - 1,
            'M4': -e3 - 1}[hyperplane_name]


def collision_search(slice_res, max_abs_root=4, verbose=True):
    hp = slice_res['hyperplane']
    m, y, z = slice_res['m'], slice_res['y'], slice_res['z']
    G = slice_res['G']

    vals = [sp.Rational(n) for n in range(-max_abs_root, max_abs_root+1)]
    hits = 0
    for r1, r2 in itertools.combinations(vals, 2):
        for r3_candidate in vals:
            if len({r1, r2, r3_candidate}) < 3:
                continue
            if invariant_condition_on_roots(hp, r1, r2, r3_candidate) != 0:
                continue
            r3 = r3_candidate

            raw = marked_states(r1, r2, r3)
            fixed = [gauge_fix(s) for s in raw]
            if any(f is None for f in fixed):
                continue

            pts, imgs = [], []
            ok = True
            for state in fixed:
                pt = invert_chart(slice_res, state)
                if pt is None:
                    ok = False
                    break
                pts.append(pt)
                subs_ = {m: pt[0], y: pt[1], z: pt[2]}
                imgs.append(tuple(sp.simplify(g.subs(subs_)) for g in G))
            if not ok:
                continue

            if len(set(pts)) == 3 and imgs[0] == imgs[1] == imgs[2]:
                hits += 1
                if verbose:
                    print(f"roots {(r1,r2,r3)}: 3 distinct points -> "
                          f"shared image {imgs[0]}")
    if verbose:
        print(f"\nTotal collisions found on slice {hp}/{NAMES[m]}: {hits}")
    return hits


if __name__ == '__main__':
    print("STAGE 1: discovering slice choice...\n")
    slice_res = discover_slice()
    if slice_res is None:
        print("No working slice found in the search space.")
    else:
        print(f"\nUsing discovered slice: {slice_res['hyperplane']}=1, "
              f"modulus={NAMES[slice_res['m']]}\n")
        print("STAGE 2: searching for collisions on this slice...\n")
        collision_search(slice_res)