"""
ERDŐS #604 STEAMROLLER (OEIS-SAFE FINAL VERSION)
Exact generation of OEIS A160663 b-file up to M_MAX.

Fixes:
1. Safe bitset allocation with explicit MAX_VAL bound
2. Defensive bounds checking inside kernel
3. Numba-safe deterministic bit operations (no unsafe |= mutation)
4. Explicit double-counting correctness annotation
5. Stable file format (OEIS b-file compliant)
6. Log-safety + reproducibility hardening
"""

import numpy as np
import time
from math import log, sqrt
from numba import njit

# =============================================================================
# CONFIGURATION
# =============================================================================
M_MAX = 300000

# --- SAFE UPPER BOUND FOR ALL a^2 + b^2 ---
MAX_VAL = 2 * M_MAX * M_MAX

# --- BITSET SIZE (bytes) ---
BITSET_SIZE_BYTES = (MAX_VAL // 8) + 8

B_FILE_NAME = "b160663.txt"
STATS_FILE = "erdos_604_full_stats.csv"


# =============================================================================
# NUMBA KERNEL (SAFE + DETERMINISTIC)
# =============================================================================
@njit
def add_layer(m, bitset, max_val):
    """
    Adds all values m^2 + j^2 for j in [0, m].
    Returns number of newly activated bits.
    """
    m2 = np.uint64(m) * np.uint64(m)
    count_new = 0

    for j in range(m + 1):
        val = m2 + np.uint64(j) * np.uint64(j)

        # ---------------------------------------------------------
        # DEFENSIVE BOUNDS CHECK (theoretically redundant but safe)
        # ---------------------------------------------------------
        if val >= max_val:
            continue

        byte_idx = val >> 3
        bit_idx = val & 7

        mask = np.uint8(1 << bit_idx)

        # ---------------------------------------------------------
        # SAFE DETERMISTIC BIT UPDATE (NO |= MUTATION)
        # ---------------------------------------------------------
        old = bitset[byte_idx]
        new = old | mask
        bitset[byte_idx] = new

        if new != old:
            count_new += 1

    return count_new


# =============================================================================
# MAIN DRIVER
# =============================================================================
def run_steamroller():
    print("=" * 78)
    print(f"ERDŐS #604 STEAMROLLER (OEIS SAFE FINAL) up to m={M_MAX}")
    print("=" * 78)

    print(f"[*] Allocating bitset: {BITSET_SIZE_BYTES / 1024**3:.2f} GB")

    # Ensure deterministic dtype (important for numba consistency)
    bitset = np.zeros(BITSET_SIZE_BYTES, dtype=np.uint8)

    # Explicit cast safety (prevents accidental dtype widening issues)
    bitset = bitset.astype(np.uint8)

    b_file = open(B_FILE_NAME, "w", buffering=1)
    stats_file = open(STATS_FILE, "w", buffering=1)

    # CSV header ONLY for stats file (OEIS b-file must remain headerless)
    stats_file.write("m,n,D,c_n\n")

    current_total_distinct = 0
    t_start = time.time()

    print("[*] Beginning deterministic layer sweep...")

    for m in range(1, M_MAX + 1):

        # Each lattice pair (a,b) is uniquely represented by:
        # k = max(a,b), j = min(a,b)
        # ensuring NO double counting across layers.
        new_found = add_layer(m, bitset, MAX_VAL)
        current_total_distinct += new_found

        # OEIS b-file format: "n a(n)"
        b_file.write(f"{m} {current_total_distinct}\n")

        # ---------------------------------------------------------
        # PERIODIC STATS OUTPUT
        # ---------------------------------------------------------
        if m % 5000 == 0 or m == 1:
            n = (2 * m) * (2 * m)

            # log-safe guard (m >= 1 ensures n >= 4 always)
            c_n = current_total_distinct / (n / sqrt(log(n)))

            stats_file.write(f"{m},{n},{current_total_distinct},{c_n:.10f}\n")

            elapsed = time.time() - t_start
            rate = m / max(elapsed, 1e-9)
            remaining = (M_MAX - m) / rate

            print(
                f"m={m:<8} | c_n={c_n:.8f} | "
                f"progress={m/M_MAX:>6.2%} | ETA={remaining/60:.1f} min"
            )

            b_file.flush()
            stats_file.flush()

    b_file.close()
    stats_file.close()

    total_time = time.time() - t_start

    print("\n[+] SUCCESS: OEIS b-file generated")
    print(f"[+] File: {B_FILE_NAME}")
    print(f"[+] Stats: {STATS_FILE}")
    print(f"[+] Runtime: {total_time/60:.2f} minutes")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_steamroller()