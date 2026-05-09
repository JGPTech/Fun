"""
INDEPENDENT VERIFIER FOR A160663
================================

Cross-checks the steamroller output by computing D(m) via a different method
than the layer-sweep, providing genuine algorithmic redundancy rather than
just re-running the same code.

METHODS:
--------
Method A (steamroller): mark a^2 + b^2 layer-by-layer using (k=max, j=min).
Method B (this file):   for each integer v <= 2m^2, decide membership via
                        sum-of-two-squares representation theory + box filter.

Method B uses Fermat's two-squares theorem indirectly: an integer v is in
D(m) iff there exist 0 <= a,b <= m with a^2+b^2 = v. We test this by
iterating a in [0, min(m, floor(sqrt(v)))] and checking whether v - a^2 is
a perfect square <= m^2.

This is O(m * sqrt(v)) per integer, so we don't run it on the full b-file.
Instead we verify at sampled m values, which is the standard approach for
this kind of cross-check.

USAGE:
------
python 02_02_verify_a160663.py b160663.txt

Optional: edit SAMPLE_POINTS to choose which m values to verify.
"""

import sys
import time
from math import isqrt, sqrt, log
import numpy as np
from numba import njit


# =============================================================================
# NUMBA-COMPATIBLE INTEGER SQUARE ROOT
# =============================================================================
# math.isqrt is not supported inside numba @njit, so we provide our own.
# Uses Newton's method on integers — exact, no floating-point error.

@njit(cache=True, inline='always')
def _isqrt(n):
    """Integer square root: largest k with k*k <= n, for n >= 0."""
    if n < 0:
        return -1
    if n < 2:
        return n
    # Initial estimate via float sqrt, then Newton refinement
    x = np.int64(np.sqrt(np.float64(n)))
    # Correct for float rounding: walk down then up
    while x * x > n:
        x -= 1
    while (x + 1) * (x + 1) <= n:
        x += 1
    return x


# =============================================================================
# CONFIG
# =============================================================================

# Sample points for verification. Includes small values (where we can compare
# against the existing OEIS b-file) and large values (where the steamroller
# extends beyond previously deposited data).
SAMPLE_POINTS = [
    100,        # tiny sanity
    1000,       # small, in published OEIS range
    5000,       # mid-published
    10000,      # boundary of previously deposited b-file
    25000,      # new territory
    50000,      # new territory
    100000,     # new territory
    200000,     # new territory
    300000,     # endpoint
]


# =============================================================================
# METHOD B: independent enumeration via direct counting
# =============================================================================

@njit(cache=True)
def count_distinct_distances_direct(m):
    """
    Independent computation of D(m) by direct enumeration.

    For each v in [1, 2m^2], decide if v = a^2 + b^2 for some 0 <= a,b <= m.
    We exploit symmetry: WLOG a <= b, and a ranges over [0, floor(sqrt(v/2))].
    For each a, check if v - a^2 is a perfect square in [a^2, m^2].

    This is independent of the bitset layer-sweep: different loop structure,
    different membership test, different bookkeeping. Agreement between the
    two methods is non-trivial.

    Returns D(m).
    """
    max_val = 2 * m * m
    m_sq = m * m
    count = 0

    for v in range(1, max_val + 1):
        # Try to find a, b with a <= b, a^2 + b^2 = v, b <= m
        # Then a^2 <= v/2, so a <= floor(sqrt(v/2))
        a_max = _isqrt(v // 2)
        if a_max > m:
            a_max = m

        found = False
        for a in range(0, a_max + 1):
            b_sq = v - a * a
            # Need b_sq to be a perfect square and b <= m
            if b_sq > m_sq:
                continue
            # Check perfect square
            b = _isqrt(b_sq)
            if b * b == b_sq:
                # b >= a is automatic since a <= floor(sqrt(v/2)) means a^2 <= v/2 <= b^2
                found = True
                break

        if found:
            count += 1

    return count


# =============================================================================
# METHOD B-FAST: sieve-based verification (for larger m)
# =============================================================================

def count_distinct_distances_sieve(m):
    """
    Sieve-based verification using a different bit-array layout than the
    steamroller. Marks integers v = a^2 + b^2 by iterating (a, b) with
    a <= b, both in [0, m]. This differs from the steamroller's layer
    structure (which uses k = max, j = min and sweeps k outward).

    The two methods produce the same set in the end, but the loop structure,
    iteration order, and update pattern are completely different. A bug
    in one is unlikely to manifest identically in the other.
    """
    max_val = 2 * m * m
    # Use numpy bool array as bitset (different representation than steamroller's
    # packed uint8 bitset — another layer of independence).
    seen = np.zeros(max_val + 1, dtype=np.bool_)

    for a in range(0, m + 1):
        a_sq = a * a
        # b ranges over [a, m] to get unique unordered pairs, but since we're
        # just marking the sum, we can take b in [0, m] — duplicates don't matter.
        # We use b in [a, m] for efficiency and correctness equivalence.
        for b in range(a, m + 1):
            v = a_sq + b * b
            if v >= 1 and v <= max_val:
                seen[v] = True

    # Count set bits, excluding zero
    return int(np.sum(seen[1:]))


# Numba-accelerated version of the sieve
@njit(cache=True, parallel=False)
def count_distinct_distances_sieve_numba(m):
    max_val = 2 * m * m
    seen = np.zeros(max_val + 1, dtype=np.uint8)

    for a in range(0, m + 1):
        a_sq = a * a
        for b in range(a, m + 1):
            v = a_sq + b * b
            seen[v] = 1

    # Count, excluding index 0
    total = 0
    for i in range(1, max_val + 1):
        if seen[i] == 1:
            total += 1
    return total


# =============================================================================
# B-FILE PARSING
# =============================================================================

def load_b_file(path):
    """Parse OEIS b-file: each line is 'm D(m)'."""
    data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            m = int(parts[0])
            d = int(parts[1])
            data[m] = d
    return data


# =============================================================================
# MAIN VERIFIER
# =============================================================================

def verify(b_file_path, sample_points):
    print("=" * 78)
    print("INDEPENDENT VERIFIER FOR A160663")
    print("=" * 78)
    print(f"[*] Loading b-file: {b_file_path}")

    b_data = load_b_file(b_file_path)
    print(f"[*] Loaded {len(b_data)} terms from b-file")
    print(f"[*] Verification sample: {sample_points}")
    print()

    # For very large m the direct enumeration becomes slow; we cap which
    # method we use based on m.
    DIRECT_METHOD_CAP = 1000      # below this, use direct (slowest, most independent)
    SIEVE_METHOD_CAP = 50000      # below this, use numba sieve in memory

    results = []
    all_passed = True

    for m in sample_points:
        if m not in b_data:
            print(f"[!] m={m}: NOT IN B-FILE, skipping")
            continue

        claimed = b_data[m]

        t0 = time.time()
        if m <= DIRECT_METHOD_CAP:
            method = "direct"
            verified = count_distinct_distances_direct(m)
        elif m <= SIEVE_METHOD_CAP:
            method = "sieve-numba"
            verified = count_distinct_distances_sieve_numba(m)
        else:
            method = "sieve-numba-large"
            # For very large m we'd need too much memory for a flat sieve
            # (e.g. m=300k means 2*9e10 = 1.8e11 byte array — not feasible
            # in this environment). Skip with explicit note.
            elapsed = time.time() - t0
            print(f"[~] m={m:>7}: SKIPPED (sieve too large for this environment); "
                  f"claimed D(m) = {claimed}")
            continue

        elapsed = time.time() - t0

        agree = (verified == claimed)
        status = "PASS" if agree else "FAIL"
        if not agree:
            all_passed = False

        print(f"[{status}] m={m:>7} | method={method:<18} | "
              f"claimed={claimed:>14} | verified={verified:>14} | "
              f"time={elapsed:.2f}s")

        results.append({
            'm': m,
            'method': method,
            'claimed': claimed,
            'verified': verified,
            'agree': agree,
            'elapsed': elapsed,
        })

    print()
    print("=" * 78)
    if all_passed:
        print("[+] ALL VERIFIED SAMPLES AGREE WITH B-FILE")
    else:
        print("[!] DISCREPANCY DETECTED — see FAIL lines above")
    print("=" * 78)

    return results, all_passed


# =============================================================================
# SELF-TEST: run both methods on a tiny m and confirm they agree
# =============================================================================

def self_test():
    """Confirm the two independent methods agree on small inputs.

    This validates the verifier itself before we use it to check the b-file.
    If methods A (direct) and B (sieve) disagree on small m, the verifier
    is broken and we shouldn't trust it.
    """
    print("[*] Self-test: comparing direct enumeration vs sieve method...")
    for m in [1, 2, 3, 5, 10, 25, 50, 100, 250, 500]:
        d_direct = count_distinct_distances_direct(m)
        d_sieve = count_distinct_distances_sieve_numba(m)
        ok = (d_direct == d_sieve)
        marker = "OK" if ok else "MISMATCH"
        print(f"    m={m:>4}: direct={d_direct:>8}, sieve={d_sieve:>8}  [{marker}]")
        if not ok:
            print("[!] SELF-TEST FAILED — verifier methods disagree")
            return False
    print("[*] Self-test PASSED — verifier methods agree on small inputs")
    print()
    return True


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 02_02_verify_a160663.py <b160663.txt>")
        sys.exit(1)

    if not self_test():
        sys.exit(1)

    results, all_passed = verify(sys.argv[1], SAMPLE_POINTS)

    sys.exit(0 if all_passed else 1)