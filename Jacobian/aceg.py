#!/usr/bin/env python3
"""
ACEG - Arbitrary Counterexample Generator
=========================================

Generate infinitely many exactly certified polynomial counterexample formulas
to the Jacobian conjecture in dimension three.

ACEG first derives the compact counterexample from the marked-factor pipeline:

    marked factorization
        -> resultant normalization
        -> affine slice
        -> residual chart
        -> induced multiplication map
        -> linear coordinate changes.

It then generates maps

    G = B o F o A,

where A and B are random compositions of elementary polynomial shears with
Jacobian determinant one.  Every generated G therefore has determinant -2,
and ACEG also transports three exact rational collision witnesses through
A^{-1}.  The program does not merely appeal to the chain rule: it expands each
generated map and recomputes its full sparse-polynomial Jacobian exactly.

Scope
-----
ACEG generates an infinite polynomial-automorphism orbit of certified
counterexamples.  It does not claim that generated maps are inequivalent under
polynomial coordinate changes, nor does it search for a new geometric
mechanism.

No third-party packages and no GPU are required.

Quick start
-----------
    python aceg.py
    python aceg.py generate --count 10 --seed 12345
    python aceg.py generate --count 3 --source-depth 3 --target-depth 3
    python aceg.py verify aceg_manifest.json
    python aceg.py base

If no command is supplied, "generate" is assumed.

Polynomial composition grows quickly with shear depth and degree.  The default
settings are intentionally conservative.  ACEG applies term, composition-work,
and Jacobian-work caps before accepting a candidate; raise them cautiously.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import random
import secrets
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]


__version__ = "1.0.0"
SCHEMA = "jgptech.aceg.manifest.v1"

Exponent = tuple[int, int, int]
Point = tuple[Fraction, Fraction, Fraction]
PolynomialMap = tuple["Poly", "Poly", "Poly"]


def Q(value: int | Fraction, denominator: int | None = None) -> Fraction:
    if denominator is None:
        return value if isinstance(value, Fraction) else Fraction(value)
    return Fraction(value, denominator)


class Poly:
    """Sparse polynomial in (x, y, t) with exact rational coefficients."""

    __slots__ = ("terms",)
    names = ("x", "y", "t")

    def __init__(self, terms: Mapping[Exponent, int | Fraction] | None = None):
        cleaned: dict[Exponent, Fraction] = {}
        for exponent, coefficient in (terms or {}).items():
            if len(exponent) != 3 or any(power < 0 for power in exponent):
                raise ValueError(f"invalid exponent: {exponent}")
            exact = Q(coefficient)
            if exact:
                cleaned[tuple(exponent)] = exact
        self.terms = cleaned

    @staticmethod
    def coerce(value: int | Fraction | "Poly") -> "Poly":
        if isinstance(value, Poly):
            return value
        exact = Q(value)
        return Poly({(0, 0, 0): exact}) if exact else Poly()

    @staticmethod
    def variable(axis: int) -> "Poly":
        if axis not in (0, 1, 2):
            raise ValueError(f"invalid variable axis: {axis}")
        exponent = [0, 0, 0]
        exponent[axis] = 1
        return Poly({tuple(exponent): Q(1)})

    @classmethod
    def from_json(cls, data: Sequence[Mapping[str, Any]]) -> "Poly":
        terms: dict[Exponent, Fraction] = {}
        for term in data:
            exponent_data = term["exponents"]
            if not isinstance(exponent_data, list) or len(exponent_data) != 3:
                raise ValueError("invalid serialized exponent")
            exponent = tuple(int(power) for power in exponent_data)
            coefficient = Fraction(
                int(term["numerator"]),
                int(term["denominator"]),
            )
            if exponent in terms:
                raise ValueError(f"duplicate serialized exponent: {exponent}")
            terms[exponent] = coefficient
        return cls(terms)

    def to_json(self) -> list[dict[str, Any]]:
        return [
            {
                "exponents": list(exponent),
                "numerator": coefficient.numerator,
                "denominator": coefficient.denominator,
            }
            for exponent, coefficient in sorted(self.terms.items())
        ]

    def __add__(self, other: int | Fraction | "Poly") -> "Poly":
        rhs = Poly.coerce(other)
        out = dict(self.terms)
        for exponent, coefficient in rhs.terms.items():
            updated = out.get(exponent, Q(0)) + coefficient
            if updated:
                out[exponent] = updated
            else:
                out.pop(exponent, None)
        return Poly(out)

    __radd__ = __add__

    def __neg__(self) -> "Poly":
        return Poly(
            {
                exponent: -coefficient
                for exponent, coefficient in self.terms.items()
            }
        )

    def __sub__(self, other: int | Fraction | "Poly") -> "Poly":
        return self + (-Poly.coerce(other))

    def __rsub__(self, other: int | Fraction | "Poly") -> "Poly":
        return Poly.coerce(other) - self

    def __mul__(self, other: int | Fraction | "Poly") -> "Poly":
        rhs = Poly.coerce(other)
        if not self.terms or not rhs.terms:
            return Poly()
        out: dict[Exponent, Fraction] = {}
        for left_exp, left_coeff in self.terms.items():
            for right_exp, right_coeff in rhs.terms.items():
                exponent = tuple(
                    left_exp[axis] + right_exp[axis] for axis in range(3)
                )
                out[exponent] = (
                    out.get(exponent, Q(0)) + left_coeff * right_coeff
                )
        return Poly(out)

    __rmul__ = __mul__

    def __pow__(self, exponent: int) -> "Poly":
        if exponent < 0:
            raise ValueError("polynomial exponent must be nonnegative")
        result = Poly.coerce(1)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            power >>= 1
            if power:
                base = base * base
        return result

    def derivative(self, axis: int) -> "Poly":
        out: dict[Exponent, Fraction] = {}
        for exponent, coefficient in self.terms.items():
            power = exponent[axis]
            if not power:
                continue
            reduced = list(exponent)
            reduced[axis] -= 1
            out[tuple(reduced)] = coefficient * power
        return Poly(out)

    def compose(self, substitutions: Sequence["Poly"]) -> "Poly":
        if len(substitutions) != 3:
            raise ValueError("expected three substitution polynomials")
        if not self.terms:
            return Poly()

        maxima = [
            max(exponent[axis] for exponent in self.terms)
            for axis in range(3)
        ]
        powers: list[list[Poly]] = []
        for axis, maximum in enumerate(maxima):
            axis_powers = [Poly.coerce(1)]
            for _ in range(maximum):
                axis_powers.append(axis_powers[-1] * substitutions[axis])
            powers.append(axis_powers)

        result = Poly()
        for exponent, coefficient in self.terms.items():
            term = Poly.coerce(coefficient)
            for axis in range(3):
                term = term * powers[axis][exponent[axis]]
            result = result + term
        return result

    def evaluate(self, point: Point) -> Fraction:
        total = Q(0)
        for exponent, coefficient in self.terms.items():
            value = coefficient
            for axis, power in enumerate(exponent):
                value *= point[axis] ** power
            total += value
        return total

    def depends_on(self, axis: int) -> bool:
        return any(exponent[axis] for exponent in self.terms)

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def total_degree(self) -> int:
        return max((sum(exponent) for exponent in self.terms), default=-1)

    def signature(self) -> tuple[tuple[Exponent, int, int], ...]:
        return tuple(
            (exponent, coefficient.numerator, coefficient.denominator)
            for exponent, coefficient in sorted(self.terms.items())
        )

    def is_constant(self, value: int | Fraction) -> bool:
        return self == Poly.coerce(value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Poly):
            return self.terms == other.terms
        if isinstance(other, (int, Fraction)):
            return self.terms == Poly.coerce(other).terms
        return NotImplemented

    def to_string(self) -> str:
        if not self.terms:
            return "0"

        def order(item: tuple[Exponent, Fraction]) -> tuple[int, Exponent]:
            exponent, _ = item
            return (-sum(exponent), tuple(-power for power in exponent))

        pieces: list[str] = []
        for exponent, coefficient in sorted(self.terms.items(), key=order):
            sign = "-" if coefficient < 0 else "+"
            magnitude = abs(coefficient)
            monomial_parts: list[str] = []
            for name, power in zip(self.names, exponent):
                if power == 1:
                    monomial_parts.append(name)
                elif power > 1:
                    monomial_parts.append(f"{name}^{power}")
            monomial = "*".join(monomial_parts)

            if magnitude.denominator == 1:
                coefficient_text = str(magnitude.numerator)
            else:
                coefficient_text = (
                    f"({magnitude.numerator}/{magnitude.denominator})"
                )

            if monomial and magnitude == 1:
                body = monomial
            elif monomial:
                body = f"{coefficient_text}*{monomial}"
            else:
                body = coefficient_text

            if not pieces:
                pieces.append(body if sign == "+" else f"-{body}")
            else:
                pieces.append(f" {sign} {body}")
        return "".join(pieces)


x, y, t = (Poly.variable(axis) for axis in range(3))
IDENTITY_MAP: PolynomialMap = (x, y, t)


class ComplexityLimitError(RuntimeError):
    """Raised when an intermediate sparse map exceeds the configured term cap."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def compose_map(
    outer: Sequence[Poly],
    inner: Sequence[Poly],
) -> PolynomialMap:
    return tuple(component.compose(inner) for component in outer)  # type: ignore[return-value]


def composition_work_estimate(
    outer: Sequence[Poly],
    inner: Sequence[Poly],
    stop_after: int | None = None,
) -> int:
    """Upper-bound raw sparse products required for outer o inner."""

    term_counts = [max(component.term_count, 1) for component in inner]
    estimate = 0
    for component in outer:
        for exponent in component.terms:
            contribution = 1
            for axis, power in enumerate(exponent):
                contribution *= term_counts[axis] ** power
            estimate += contribution
            if stop_after is not None and estimate > stop_after:
                return estimate
    return estimate


def evaluate_map(polynomial_map: Sequence[Poly], point: Point) -> Point:
    return tuple(component.evaluate(point) for component in polynomial_map)  # type: ignore[return-value]


def det3(matrix: Sequence[Sequence[Poly]]) -> Poly:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


def jacobian_determinant(polynomial_map: Sequence[Poly]) -> Poly:
    matrix = [
        [component.derivative(axis) for axis in range(3)]
        for component in polynomial_map
    ]
    return det3(matrix)


def jacobian_work_estimate(polynomial_map: Sequence[Poly]) -> int:
    """Estimate sparse multiply pairs in the six determinant triple-products."""

    counts = [
        [
            component.derivative(axis).term_count
            for axis in range(3)
        ]
        for component in polynomial_map
    ]
    a, b, c = counts[0]
    d, e, f = counts[1]
    g, h, i = counts[2]
    return (
        a * e * i
        + a * f * h
        + b * d * i
        + b * f * g
        + c * d * h
        + c * e * g
    )


def map_hash(polynomial_map: Sequence[Poly]) -> str:
    canonical = repr(
        tuple(component.signature() for component in polynomial_map)
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def point_to_json(point: Point) -> list[str]:
    return [str(coordinate) for coordinate in point]


def point_from_json(data: Sequence[str]) -> Point:
    if len(data) != 3:
        raise ValueError("serialized point must have three coordinates")
    return tuple(Fraction(coordinate) for coordinate in data)  # type: ignore[return-value]


def derive_pipeline_map() -> PolynomialMap:
    """Derive F from Phi, multiplication M, and the linear maps T and S."""

    a, chart_y, z = x, y, t
    b = 1 + a * chart_y
    c = 1 - Q(3, 2) * a * chart_y + a**2 * z
    d = (
        Q(1, 2) * chart_y
        - a * z
        + Q(3, 2) * a * chart_y**2
        - a**2 * chart_y * z
    )
    e = (
        -2 * z
        + 4 * chart_y**2
        - 4 * a * chart_y * z
        + 3 * a * chart_y**3
        - 2 * a**2 * chart_y**2 * z
    )

    resultant = a**2 * e - a * b * d + b**2 * c
    slice_equation = a * d + b * c
    inverse_y = 2 * b * d - a * e
    if not resultant.is_constant(1):
        raise AssertionError("pipeline chart failed resultant normalization")
    if not slice_equation.is_constant(1):
        raise AssertionError("pipeline chart failed affine slice")
    if inverse_y != chart_y:
        raise AssertionError("pipeline chart failed first inverse coordinate")

    induced = (a * c, a * e + b * d, b * e)
    source_change = (x, y, -Q(1, 2) * t)
    transformed = compose_map(induced, source_change)
    compact: PolynomialMap = (
        transformed[2],
        2 * transformed[1],
        2 * transformed[0],
    )

    # Regression comparison only: the generator source is the pipeline
    # derivation above.
    u = 1 + x * y
    expected: PolynomialMap = (
        u**3 * t + y**2 * u * (4 + 3 * x * y),
        y + 3 * x * u**2 * t + 3 * x * y**2 * (4 + 3 * x * y),
        2 * x - 3 * x**2 * y - x**3 * t,
    )
    if compact != expected:
        raise AssertionError(
            "pipeline derivation does not match the compact certificate"
        )
    if not jacobian_determinant(compact).is_constant(-2):
        raise AssertionError("base pipeline map does not have determinant -2")
    return compact


BASE_POINTS: tuple[Point, Point, Point] = (
    (Q(0), Q(0), Q(-1, 4)),
    (Q(1), Q(-3, 2), Q(13, 2)),
    (Q(-1), Q(3, 2), Q(13, 2)),
)
BASE_IMAGE: Point = (Q(-1, 4), Q(0), Q(0))


@dataclass(frozen=True)
class Shear:
    """Elementary automorphism q_axis <- q_axis + polynomial(other axes)."""

    axis: int
    polynomial: Poly

    def validate(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError(f"invalid shear axis: {self.axis}")
        if not self.polynomial.terms:
            raise ValueError("zero shear is not allowed")
        if self.polynomial.depends_on(self.axis):
            raise ValueError(
                "shear polynomial must not depend on its modified coordinate"
            )

    def elementary_map(self) -> PolynomialMap:
        self.validate()
        components = [x, y, t]
        components[self.axis] = components[self.axis] + self.polynomial
        return tuple(components)  # type: ignore[return-value]

    def apply_to_point(self, point: Point) -> Point:
        self.validate()
        updated = list(point)
        updated[self.axis] += self.polynomial.evaluate(point)
        return tuple(updated)  # type: ignore[return-value]

    def apply_inverse_to_point(self, point: Point) -> Point:
        self.validate()
        updated = list(point)
        updated[self.axis] -= self.polynomial.evaluate(point)
        return tuple(updated)  # type: ignore[return-value]

    def to_json(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "axis_name": Poly.names[self.axis],
            "polynomial": self.polynomial.to_json(),
            "expanded": self.polynomial.to_string(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Shear":
        shear = cls(
            axis=int(data["axis"]),
            polynomial=Poly.from_json(data["polynomial"]),
        )
        shear.validate()
        return shear


def monomial_pool(axis: int, max_degree: int) -> list[Exponent]:
    pool: list[Exponent] = []
    for ex in range(max_degree + 1):
        for ey in range(max_degree + 1):
            for et in range(max_degree + 1):
                exponent = (ex, ey, et)
                degree = sum(exponent)
                if exponent[axis] == 0 and 1 <= degree <= max_degree:
                    pool.append(exponent)
    return pool


def nonzero_integer(rng: random.Random, bound: int) -> int:
    value = 0
    while value == 0:
        value = rng.randint(-bound, bound)
    return value


def random_shear(
    rng: random.Random,
    max_degree: int,
    requested_terms: int,
    coefficient_bound: int,
) -> Shear:
    axis = rng.randrange(3)
    nonconstant_pool = monomial_pool(axis, max_degree)
    if not nonconstant_pool:
        raise ValueError("max shear degree must be at least one")

    # At least one nonconstant monomial makes the operation a genuine shear.
    selected = [rng.choice(nonconstant_pool)]
    remaining_pool = [
        exponent
        for exponent in [(0, 0, 0), *nonconstant_pool]
        if exponent not in selected
    ]
    additional = min(max(requested_terms - 1, 0), len(remaining_pool))
    selected.extend(rng.sample(remaining_pool, additional))

    polynomial = Poly(
        {
            exponent: nonzero_integer(rng, coefficient_bound)
            for exponent in selected
        }
    )
    shear = Shear(axis=axis, polynomial=polynomial)
    shear.validate()
    return shear


def random_shear_sequence(
    rng: random.Random,
    depth: int,
    max_degree: int,
    shear_terms: int,
    coefficient_bound: int,
) -> tuple[Shear, ...]:
    return tuple(
        random_shear(
            rng,
            max_degree=max_degree,
            requested_terms=shear_terms,
            coefficient_bound=coefficient_bound,
        )
        for _ in range(depth)
    )


def transformation_map(
    operations: Sequence[Shear],
    term_cap: int | None = None,
) -> PolynomialMap:
    current = IDENTITY_MAP
    for operation in operations:
        current = compose_map(operation.elementary_map(), current)
        if (
            term_cap is not None
            and max(component.term_count for component in current) > term_cap
        ):
            raise ComplexityLimitError(
                "term_cap",
                "intermediate automorphism exceeded the term cap"
            )
    return current


def orbit_map(
    base_map: PolynomialMap,
    source_operations: Sequence[Shear],
    target_operations: Sequence[Shear],
    term_cap: int | None = None,
    composition_work_cap: int | None = None,
) -> PolynomialMap:
    """Build B o F o A one elementary shear at a time."""

    current = base_map

    # If A = E_k o ... o E_1, then F o A is assembled from E_k backward.
    for operation in reversed(source_operations):
        elementary = operation.elementary_map()
        if (
            composition_work_cap is not None
            and composition_work_estimate(
                current,
                elementary,
                stop_after=composition_work_cap,
            )
            > composition_work_cap
        ):
            raise ComplexityLimitError(
                "composition_work_cap",
                "source composition exceeded the work cap"
            )
        current = compose_map(current, elementary)
        if (
            term_cap is not None
            and max(component.term_count for component in current) > term_cap
        ):
            raise ComplexityLimitError(
                "term_cap",
                "source-precomposed map exceeded the term cap"
            )

    # If B = E_k o ... o E_1, apply the target shears in forward order.
    for operation in target_operations:
        elementary = operation.elementary_map()
        if (
            composition_work_cap is not None
            and composition_work_estimate(
                elementary,
                current,
                stop_after=composition_work_cap,
            )
            > composition_work_cap
        ):
            raise ComplexityLimitError(
                "composition_work_cap",
                "target composition exceeded the work cap"
            )
        current = compose_map(elementary, current)
        if (
            term_cap is not None
            and max(component.term_count for component in current) > term_cap
        ):
            raise ComplexityLimitError(
                "term_cap",
                "target-postcomposed map exceeded the term cap"
            )
    return current


def apply_operations_to_point(
    operations: Sequence[Shear],
    point: Point,
) -> Point:
    current = point
    for operation in operations:
        current = operation.apply_to_point(current)
    return current


def apply_inverse_operations_to_point(
    operations: Sequence[Shear],
    point: Point,
) -> Point:
    current = point
    for operation in reversed(operations):
        current = operation.apply_inverse_to_point(current)
    return current


def serialize_map(
    polynomial_map: Sequence[Poly],
    include_expanded: bool,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for name, component in zip(("F1", "F2", "F3"), polynomial_map):
        record: dict[str, Any] = {
            "name": name,
            "degree": component.total_degree,
            "term_count": component.term_count,
            "terms": component.to_json(),
        }
        if include_expanded:
            record["expanded"] = component.to_string()
        serialized.append(record)
    return serialized


def deserialize_map(data: Sequence[Mapping[str, Any]]) -> PolynomialMap:
    if len(data) != 3:
        raise ValueError("serialized map must have three coordinates")
    return tuple(Poly.from_json(component["terms"]) for component in data)  # type: ignore[return-value]


def peak_rss_mib() -> float | None:
    """Return peak resident memory on Linux/macOS/Windows when available."""

    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            success = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            if success:
                return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
        except (AttributeError, OSError):
            return None
        return None

    if resource is None:
        return None
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return maximum / (1024.0 * 1024.0)
    return maximum / 1024.0


def build_candidate(
    base_map: PolynomialMap,
    source_operations: Sequence[Shear],
    target_operations: Sequence[Shear],
    term_cap: int,
    composition_work_cap: int,
    jacobian_work_cap: int,
    include_expanded: bool,
    index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        generated = orbit_map(
            base_map,
            source_operations,
            target_operations,
            term_cap=term_cap,
            composition_work_cap=composition_work_cap,
        )
    except ComplexityLimitError as exc:
        return None, exc.reason

    term_counts = [component.term_count for component in generated]
    if max(term_counts) > term_cap:
        return None, "term_cap"

    work_estimate = jacobian_work_estimate(generated)
    if work_estimate > jacobian_work_cap:
        return None, "jacobian_work_cap"

    determinant = jacobian_determinant(generated)
    if not determinant.is_constant(-2):
        return None, "jacobian"

    transported_points = tuple(
        apply_inverse_operations_to_point(source_operations, point)
        for point in BASE_POINTS
    )
    expected_image = apply_operations_to_point(target_operations, BASE_IMAGE)
    images = tuple(
        evaluate_map(generated, point) for point in transported_points
    )
    if len(set(transported_points)) != 3:
        return None, "witness_distinctness"
    if not all(image == expected_image for image in images):
        return None, "collision"

    canonical_hash = map_hash(generated)
    record = {
        "index": index,
        "id": f"ACEG-{index:04d}-{canonical_hash[:12]}",
        "map_sha256": canonical_hash,
        "source_automorphism": [
            operation.to_json() for operation in source_operations
        ],
        "target_automorphism": [
            operation.to_json() for operation in target_operations
        ],
        "map": serialize_map(generated, include_expanded=include_expanded),
        "jacobian_work_estimate": work_estimate,
        "jacobian_determinant": "-2",
        "collision_preimages": [
            point_to_json(point) for point in transported_points
        ],
        "collision_image": point_to_json(expected_image),
        "verified": True,
    }
    return record, None


def validate_generation_settings(args: argparse.Namespace) -> None:
    positive = {
        "--count": args.count,
        "--max-shear-degree": args.max_shear_degree,
        "--shear-terms": args.shear_terms,
        "--coefficient-bound": args.coefficient_bound,
        "--term-cap": args.term_cap,
        "--composition-work-cap": args.composition_work_cap,
        "--jacobian-work-cap": args.jacobian_work_cap,
        "--attempt-cap": args.attempt_cap,
    }
    for name, value in positive.items():
        if value < 1:
            raise SystemExit(f"{name} must be positive")
    if args.source_depth < 0 or args.target_depth < 0:
        raise SystemExit("automorphism depths must be nonnegative")
    if args.source_depth == 0 and args.target_depth == 0 and args.count > 1:
        raise SystemExit(
            "at least one automorphism depth must be positive when count > 1"
        )


def generate_manifest(args: argparse.Namespace) -> dict[str, Any]:
    validate_generation_settings(args)
    start = time.perf_counter()
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    rng = random.Random(seed)

    base_map = derive_pipeline_map()
    if any(
        evaluate_map(base_map, point) != BASE_IMAGE for point in BASE_POINTS
    ):
        raise AssertionError("base collision certificate failed")

    maps: list[dict[str, Any]] = []
    hashes: set[str] = set()
    rejected = {
        "duplicate": 0,
        "term_cap": 0,
        "composition_work_cap": 0,
        "jacobian_work_cap": 0,
        "jacobian": 0,
        "witness_distinctness": 0,
        "collision": 0,
    }

    attempts = 0
    while len(maps) < args.count and attempts < args.attempt_cap:
        attempts += 1
        source_operations = random_shear_sequence(
            rng,
            depth=args.source_depth,
            max_degree=args.max_shear_degree,
            shear_terms=args.shear_terms,
            coefficient_bound=args.coefficient_bound,
        )
        target_operations = random_shear_sequence(
            rng,
            depth=args.target_depth,
            max_degree=args.max_shear_degree,
            shear_terms=args.shear_terms,
            coefficient_bound=args.coefficient_bound,
        )
        record, reason = build_candidate(
            base_map=base_map,
            source_operations=source_operations,
            target_operations=target_operations,
            term_cap=args.term_cap,
            composition_work_cap=args.composition_work_cap,
            jacobian_work_cap=args.jacobian_work_cap,
            include_expanded=not args.compact,
            index=len(maps),
        )
        if record is None:
            rejected[reason or "jacobian"] += 1
            continue
        candidate_hash = record["map_sha256"]
        if candidate_hash in hashes:
            rejected["duplicate"] += 1
            continue
        hashes.add(candidate_hash)
        maps.append(record)

    if len(maps) != args.count:
        raise RuntimeError(
            f"generated {len(maps)} of {args.count} requested maps after "
            f"{attempts} attempts; raise --attempt-cap or relax complexity"
        )

    elapsed = time.perf_counter() - start
    peak = peak_rss_mib()
    largest_terms = max(
        component["term_count"]
        for record in maps
        for component in record["map"]
    )
    largest_degree = max(
        component["degree"]
        for record in maps
        for component in record["map"]
    )

    return {
        "schema": SCHEMA,
        "generator": "ACEG - Arbitrary Counterexample Generator",
        "version": __version__,
        "scope": (
            "Exact counterexamples in the polynomial-automorphism orbit of "
            "the marked-factor pipeline map; no inequivalence claim."
        ),
        "seed": seed,
        "settings": {
            "count": args.count,
            "source_depth": args.source_depth,
            "target_depth": args.target_depth,
            "max_shear_degree": args.max_shear_degree,
            "shear_terms": args.shear_terms,
            "coefficient_bound": args.coefficient_bound,
            "term_cap": args.term_cap,
            "composition_work_cap": args.composition_work_cap,
            "jacobian_work_cap": args.jacobian_work_cap,
            "attempt_cap": args.attempt_cap,
            "expanded_formulas_included": not args.compact,
        },
        "pipeline": {
            "base_map_sha256": map_hash(base_map),
            "base_map": serialize_map(
                base_map,
                include_expanded=not args.compact,
            ),
            "base_jacobian_determinant": "-2",
            "base_collision_preimages": [
                point_to_json(point) for point in BASE_POINTS
            ],
            "base_collision_image": point_to_json(BASE_IMAGE),
        },
        "summary": {
            "generated": len(maps),
            "attempts": attempts,
            "rejected": rejected,
            "all_verified": all(record["verified"] for record in maps),
            "all_hashes_distinct": len(hashes) == len(maps),
            "largest_coordinate_degree": largest_degree,
            "largest_coordinate_terms": largest_terms,
            "elapsed_seconds": round(elapsed, 6),
            "peak_rss_mib": round(peak, 3) if peak is not None else None,
        },
        "maps": maps,
    }


def verify_manifest_data(manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    map_results: list[dict[str, Any]] = []

    if manifest.get("schema") != SCHEMA:
        errors.append(
            f"unsupported schema: {manifest.get('schema')!r}; expected {SCHEMA!r}"
        )

    try:
        base_map = derive_pipeline_map()
        pipeline = manifest["pipeline"]
        stored_base = deserialize_map(pipeline["base_map"])
        if stored_base != base_map:
            errors.append("stored base map does not match pipeline derivation")
        if pipeline["base_map_sha256"] != map_hash(base_map):
            errors.append("stored base map hash is invalid")
    except (KeyError, TypeError, ValueError, AssertionError) as exc:
        errors.append(f"base pipeline record is invalid: {exc}")
        base_map = derive_pipeline_map()

    for position, record in enumerate(manifest.get("maps", [])):
        local_errors: list[str] = []
        try:
            source_operations = tuple(
                Shear.from_json(item)
                for item in record["source_automorphism"]
            )
            target_operations = tuple(
                Shear.from_json(item)
                for item in record["target_automorphism"]
            )
            stored_map = deserialize_map(record["map"])
            rebuilt_map = orbit_map(
                base_map,
                source_operations,
                target_operations,
            )
            if stored_map != rebuilt_map:
                local_errors.append(
                    "stored map does not match its recorded automorphisms"
                )

            calculated_hash = map_hash(stored_map)
            if calculated_hash != record["map_sha256"]:
                local_errors.append("map hash mismatch")

            determinant = jacobian_determinant(stored_map)
            if not determinant.is_constant(-2):
                local_errors.append(
                    f"Jacobian is {determinant.to_string()}, not -2"
                )

            stored_points = tuple(
                point_from_json(point)
                for point in record["collision_preimages"]
            )
            rebuilt_points = tuple(
                apply_inverse_operations_to_point(
                    source_operations,
                    point,
                )
                for point in BASE_POINTS
            )
            if stored_points != rebuilt_points:
                local_errors.append("collision witnesses were not transported correctly")
            if len(set(stored_points)) != 3:
                local_errors.append("collision witnesses are not distinct")

            expected_image = apply_operations_to_point(
                target_operations,
                BASE_IMAGE,
            )
            stored_image = point_from_json(record["collision_image"])
            if stored_image != expected_image:
                local_errors.append("stored collision image is incorrect")
            images = tuple(
                evaluate_map(stored_map, point) for point in stored_points
            )
            if not all(image == expected_image for image in images):
                local_errors.append("collision substitution failed")
        except (KeyError, TypeError, ValueError, AssertionError) as exc:
            local_errors.append(f"invalid record: {exc}")

        map_results.append(
            {
                "position": position,
                "id": record.get("id", f"map-{position}"),
                "passed": not local_errors,
                "errors": local_errors,
            }
        )

    if not manifest.get("maps"):
        errors.append("manifest contains no generated maps")
    if any(not result["passed"] for result in map_results):
        errors.append("one or more generated maps failed verification")

    return {
        "passed": not errors,
        "manifest_errors": errors,
        "maps_checked": len(map_results),
        "map_results": map_results,
    }


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def print_generation_summary(
    manifest: Mapping[str, Any],
    output: Path,
) -> None:
    summary = manifest["summary"]
    print("ACEG generation complete")
    print(f"seed: {manifest['seed']}")
    print(f"generated: {summary['generated']}")
    print(f"all_verified: {summary['all_verified']}")
    print(f"all_hashes_distinct: {summary['all_hashes_distinct']}")
    print(f"rejected_attempts: {summary['rejected']}")
    print(f"largest_coordinate_degree: {summary['largest_coordinate_degree']}")
    print(f"largest_coordinate_terms: {summary['largest_coordinate_terms']}")
    print(f"elapsed_seconds: {summary['elapsed_seconds']}")
    print(f"peak_rss_mib: {summary['peak_rss_mib']}")
    print(f"manifest: {output.resolve()}")


def command_generate(args: argparse.Namespace) -> int:
    manifest = generate_manifest(args)
    write_json(args.output, manifest)
    verification = verify_manifest_data(manifest)
    if not verification["passed"]:
        raise AssertionError(
            "post-write manifest verification failed: "
            + "; ".join(verification["manifest_errors"])
        )
    if not args.quiet:
        print_generation_summary(manifest, args.output)
    return 0


def command_verify(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = verify_manifest_data(manifest)
    print(f"manifest: {args.manifest.resolve()}")
    print(f"passed: {result['passed']}")
    print(f"maps_checked: {result['maps_checked']}")
    for record in result["map_results"]:
        print(f"{record['id']}: {'PASS' if record['passed'] else 'FAIL'}")
        for error in record["errors"]:
            print(f"  - {error}")
    for error in result["manifest_errors"]:
        print(f"manifest error: {error}")
    return 0 if result["passed"] else 1


def command_base(_: argparse.Namespace) -> int:
    base_map = derive_pipeline_map()
    print("Pipeline-derived base counterexample")
    for name, component in zip(("F1", "F2", "F3"), base_map):
        print(f"{name} = {component.to_string()}")
    print(f"determinant = {jacobian_determinant(base_map).to_string()}")
    print("collision preimages:")
    for point in BASE_POINTS:
        print(f"  {point_to_json(point)}")
    print(f"collision image: {point_to_json(BASE_IMAGE)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate exactly certified Jacobian counterexamples.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="generate and exactly verify counterexample maps",
    )
    generate.add_argument("--count", type=int, default=5)
    generate.add_argument(
        "--seed",
        type=int,
        default=None,
        help="integer seed; omitted selects and records a random 64-bit seed",
    )
    generate.add_argument("--source-depth", type=int, default=2)
    generate.add_argument("--target-depth", type=int, default=2)
    generate.add_argument("--max-shear-degree", type=int, default=2)
    generate.add_argument("--shear-terms", type=int, default=2)
    generate.add_argument("--coefficient-bound", type=int, default=3)
    generate.add_argument("--term-cap", type=int, default=5000)
    generate.add_argument(
        "--composition-work-cap",
        type=int,
        default=5_000_000,
        help=(
            "reject a shear before composition when its sparse-product "
            "estimate exceeds this limit"
        ),
    )
    generate.add_argument(
        "--jacobian-work-cap",
        type=int,
        default=10_000_000,
        help=(
            "reject candidates whose estimated determinant expansion exceeds "
            "this many sparse coefficient products"
        ),
    )
    generate.add_argument("--attempt-cap", type=int, default=100)
    generate.add_argument(
        "--compact",
        action="store_true",
        help="omit human-readable expanded formulas from the JSON",
    )
    generate.add_argument(
        "--output",
        type=Path,
        default=Path("aceg_manifest.json"),
    )
    generate.add_argument("--quiet", action="store_true")
    generate.set_defaults(handler=command_generate)

    verify = subparsers.add_parser(
        "verify",
        help="rebuild and exactly verify an ACEG manifest",
    )
    verify.add_argument("manifest", type=Path)
    verify.set_defaults(handler=command_verify)

    base = subparsers.add_parser(
        "base",
        help="display the pipeline-derived base certificate",
    )
    base.set_defaults(handler=command_base)
    return parser


def normalize_argv(argv: Iterable[str] | None) -> list[str]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {"generate", "verify", "base"}
    if not arguments:
        return ["generate"]
    if arguments[0] not in commands and arguments[0] not in {
        "-h",
        "--help",
        "--version",
    }:
        return ["generate", *arguments]
    return arguments


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(normalize_argv(argv))
    try:
        return args.handler(args)
    except (AssertionError, OSError, RuntimeError, ValueError) as exc:
        print(f"ACEG error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
