#!/usr/bin/env julia

#=
ACEG - Arbitrary Counterexample Generator (Julia sister edition)
================================================================

ACEG derives the compact three-dimensional Jacobian counterexample from the
marked-factor pipeline and generates infinitely many exactly certified formulas

    G = B o F o A,

where A and B are compositions of elementary determinant-one polynomial
shears. Every generated map is expanded over Rational{BigInt}; its full
Jacobian determinant is recomputed exactly, and three rational collision
witnesses are transported through A^(-1) and checked by substitution.

This Julia edition uses only Julia standard libraries. It reads and writes the
same jgptech.aceg.manifest.v1 JSON schema as aceg.py and reproduces Python's
canonical polynomial SHA-256 hashes.

Scope: ACEG generates the polynomial-automorphism orbit of the pipeline map.
It does not claim that its outputs are inequivalent under coordinate changes.
Seeds reproduce Julia runs; Julia and Python intentionally use their native
RNG streams, so equal seeds need not select equal orbit representatives.

Quick start
-----------
    julia aceg.jl selftest aceg_sample_manifest.json
    julia aceg.jl
    julia aceg.jl generate --count 10 --seed 12345
    julia aceg.jl verify aceg_sample_manifest.json
    julia aceg.jl base

If no command is supplied, "generate" is assumed. Complexity grows quickly
with shear depth and degree; conservative term and work caps are applied.
=#

using Random
using SHA
using Printf

const ACEG_VERSION = "1.0.1"
const ACEG_SCHEMA = "jgptech.aceg.manifest.v1"
const PYTHON_BASE_MAP_SHA256 =
    "ce70ce88ad5ef1553386ebcfc9ff5b4b1c6d7b239defc514cb66c41bc07423c7"
const Exponent = NTuple{3, Int}
const Rat = Rational{BigInt}
const Point = NTuple{3, Rat}

rat(n::Integer) = BigInt(n) // BigInt(1)
rat(n::Integer, d::Integer) = BigInt(n) // BigInt(d)
asrat(n::Integer) = rat(n)
asrat(q::Rat) = q


# ---------------------------------------------------------------------------
# Exact sparse polynomial arithmetic
# ---------------------------------------------------------------------------

struct Poly
    terms::Dict{Exponent, Rat}
end

function poly(raw::AbstractDict = Dict{Exponent, Rat}())
    cleaned = Dict{Exponent, Rat}()
    for (exponent, coefficient_raw) in raw
        length(exponent) == 3 || error("invalid exponent: $exponent")
        all(power -> power >= 0, exponent) ||
            error("negative exponent: $exponent")
        coefficient = asrat(coefficient_raw)
        coefficient == 0 && continue
        cleaned[Tuple(exponent)] = coefficient
    end
    return Poly(cleaned)
end

function pconst(value::Union{Integer, Rat})
    coefficient = asrat(value)
    coefficient == 0 && return poly()
    return poly(Dict{Exponent, Rat}((0, 0, 0) => coefficient))
end

function pvar(axis::Int)
    1 <= axis <= 3 || error("invalid variable axis: $axis")
    exponent = ntuple(i -> i == axis ? 1 : 0, 3)
    return poly(Dict{Exponent, Rat}(exponent => rat(1)))
end

function Base.:+(left::Poly, right::Poly)
    out = copy(left.terms)
    for (exponent, coefficient) in right.terms
        updated = get(out, exponent, rat(0)) + coefficient
        if updated == 0
            pop!(out, exponent, nothing)
        else
            out[exponent] = updated
        end
    end
    return poly(out)
end

Base.:+(left::Poly, right::Union{Integer, Rat}) = left + pconst(right)
Base.:+(left::Union{Integer, Rat}, right::Poly) = pconst(left) + right
Base.:-(value::Poly) =
    poly(Dict(exponent => -coefficient for
              (exponent, coefficient) in value.terms))
Base.:-(left::Poly, right::Poly) = left + (-right)
Base.:-(left::Poly, right::Union{Integer, Rat}) = left - pconst(right)
Base.:-(left::Union{Integer, Rat}, right::Poly) = pconst(left) - right

function Base.:*(left::Poly, right::Poly)
    (isempty(left.terms) || isempty(right.terms)) && return poly()
    out = Dict{Exponent, Rat}()
    for (left_exp, left_coeff) in left.terms
        for (right_exp, right_coeff) in right.terms
            exponent = ntuple(
                axis -> left_exp[axis] + right_exp[axis],
                3,
            )
            out[exponent] =
                get(out, exponent, rat(0)) + left_coeff * right_coeff
        end
    end
    return poly(out)
end

Base.:*(left::Poly, right::Union{Integer, Rat}) = left * pconst(right)
Base.:*(left::Union{Integer, Rat}, right::Poly) = pconst(left) * right

function Base.:^(value::Poly, exponent::Integer)
    exponent >= 0 || error("polynomial exponent must be nonnegative")
    result = pconst(1)
    base = value
    power = Int(exponent)
    while power > 0
        isodd(power) && (result = result * base)
        power >>= 1
        power > 0 && (base = base * base)
    end
    return result
end

Base.:(==)(left::Poly, right::Poly) = left.terms == right.terms

function derivative(value::Poly, axis::Int)
    out = Dict{Exponent, Rat}()
    for (exponent, coefficient) in value.terms
        power = exponent[axis]
        power == 0 && continue
        reduced = ntuple(
            index -> index == axis ? exponent[index] - 1 : exponent[index],
            3,
        )
        out[reduced] = coefficient * power
    end
    return poly(out)
end

function compose_poly(value::Poly, substitutions::NTuple{3, Poly})
    isempty(value.terms) && return poly()

    maxima = zeros(Int, 3)
    for exponent in keys(value.terms)
        for axis in 1:3
            maxima[axis] = max(maxima[axis], exponent[axis])
        end
    end

    powers = Vector{Vector{Poly}}(undef, 3)
    for axis in 1:3
        axis_powers = Poly[pconst(1)]
        for _ in 1:maxima[axis]
            push!(axis_powers, axis_powers[end] * substitutions[axis])
        end
        powers[axis] = axis_powers
    end

    result = poly()
    for (exponent, coefficient) in value.terms
        term = pconst(coefficient)
        for axis in 1:3
            term = term * powers[axis][exponent[axis] + 1]
        end
        result = result + term
    end
    return result
end

function evaluate_poly(value::Poly, point::Point)
    total = rat(0)
    for (exponent, coefficient) in value.terms
        term_value = coefficient
        for axis in 1:3
            term_value *= point[axis]^exponent[axis]
        end
        total += term_value
    end
    return total
end

depends_on(value::Poly, axis::Int) =
    any(exponent -> exponent[axis] != 0, keys(value.terms))
term_count(value::Poly) = length(value.terms)
total_degree(value::Poly) =
    isempty(value.terms) ? -1 : maximum(sum(exponent) for exponent in keys(value.terms))
is_constant(value::Poly, expected::Union{Integer, Rat}) =
    value == pconst(expected)

const XVAR = pvar(1)
const YVAR = pvar(2)
const TVAR = pvar(3)
const IDENTITY_MAP = (XVAR, YVAR, TVAR)
const PolynomialMap = NTuple{3, Poly}

compose_map(outer::NTuple{3, Poly}, inner::NTuple{3, Poly}) =
    ntuple(index -> compose_poly(outer[index], inner), 3)

evaluate_map(polynomial_map::NTuple{3, Poly}, point::Point) =
    ntuple(index -> evaluate_poly(polynomial_map[index], point), 3)

function determinant3(matrix::Matrix{Poly})
    a, b, c = matrix[1, 1], matrix[1, 2], matrix[1, 3]
    d, e, f = matrix[2, 1], matrix[2, 2], matrix[2, 3]
    g, h, i = matrix[3, 1], matrix[3, 2], matrix[3, 3]
    return a * (e * i - f * h) -
           b * (d * i - f * g) +
           c * (d * h - e * g)
end

function jacobian_determinant(polynomial_map::NTuple{3, Poly})
    matrix = Matrix{Poly}(undef, 3, 3)
    for row in 1:3, column in 1:3
        matrix[row, column] = derivative(polynomial_map[row], column)
    end
    return determinant3(matrix)
end

function composition_work_estimate(
    outer::NTuple{3, Poly},
    inner::NTuple{3, Poly};
    stop_after::Union{Nothing, Int} = nothing,
)
    counts = ntuple(index -> max(term_count(inner[index]), 1), 3)
    estimate = BigInt(0)
    for component in outer
        for exponent in keys(component.terms)
            contribution = BigInt(1)
            for axis in 1:3
                contribution *= BigInt(counts[axis])^exponent[axis]
            end
            estimate += contribution
            if stop_after !== nothing && estimate > stop_after
                return estimate
            end
        end
    end
    return estimate
end

function jacobian_work_estimate(polynomial_map::NTuple{3, Poly})
    counts = [
        term_count(derivative(polynomial_map[row], column))
        for row in 1:3, column in 1:3
    ]
    a, b, c = counts[1, 1], counts[1, 2], counts[1, 3]
    d, e, f = counts[2, 1], counts[2, 2], counts[2, 3]
    g, h, i = counts[3, 1], counts[3, 2], counts[3, 3]
    return BigInt(a) * e * i +
           BigInt(a) * f * h +
           BigInt(b) * d * i +
           BigInt(b) * f * g +
           BigInt(c) * d * h +
           BigInt(c) * e * g
end


# ---------------------------------------------------------------------------
# Human-readable formulas and Python-compatible canonical hashes
# ---------------------------------------------------------------------------

function polynomial_string(value::Poly)
    isempty(value.terms) && return "0"
    entries = collect(value.terms)
    sort!(
        entries;
        by = pair -> (
            -sum(first(pair)),
            ntuple(axis -> -first(pair)[axis], 3),
        ),
    )

    pieces = String[]
    names = ("x", "y", "t")
    for (position, pair) in enumerate(entries)
        exponent, coefficient = first(pair), last(pair)
        sign = coefficient < 0 ? "-" : "+"
        magnitude = abs(coefficient)
        monomial_parts = String[]
        for axis in 1:3
            power = exponent[axis]
            power == 1 && push!(monomial_parts, names[axis])
            power > 1 && push!(monomial_parts, "$(names[axis])^$power")
        end
        monomial = join(monomial_parts, "*")
        coefficient_text = denominator(magnitude) == 1 ?
            string(numerator(magnitude)) :
            "($(numerator(magnitude))/$(denominator(magnitude)))"

        body = if !isempty(monomial) && magnitude == 1
            monomial
        elseif !isempty(monomial)
            "$coefficient_text*$monomial"
        else
            coefficient_text
        end

        if position == 1
            push!(pieces, sign == "+" ? body : "-$body")
        else
            push!(pieces, " $sign $body")
        end
    end
    return join(pieces)
end

function python_tuple(items::Vector{String})
    isempty(items) && return "()"
    length(items) == 1 && return "(" * items[1] * ",)"
    return "(" * join(items, ", ") * ")"
end

function python_exponent_repr(exponent::Exponent)
    return "(" * join(string.(collect(exponent)), ", ") * ")"
end

function python_poly_signature(value::Poly)
    terms = String[]
    for exponent in sort!(collect(keys(value.terms)))
        coefficient = value.terms[exponent]
        push!(
            terms,
            python_tuple([
                python_exponent_repr(exponent),
                string(numerator(coefficient)),
                string(denominator(coefficient)),
            ]),
        )
    end
    return python_tuple(terms)
end

function map_hash(polynomial_map::NTuple{3, Poly})
    canonical = python_tuple([
        python_poly_signature(component) for component in polynomial_map
    ])
    return bytes2hex(sha256(codeunits(canonical)))
end


# ---------------------------------------------------------------------------
# Pipeline derivation and base certificate
# ---------------------------------------------------------------------------

function derive_pipeline_map()
    a, chart_y, z = XVAR, YVAR, TVAR
    b = 1 + a * chart_y
    c = 1 - rat(3, 2) * a * chart_y + a^2 * z
    d = rat(1, 2) * chart_y -
        a * z +
        rat(3, 2) * a * chart_y^2 -
        a^2 * chart_y * z
    e = -2 * z +
        4 * chart_y^2 -
        4 * a * chart_y * z +
        3 * a * chart_y^3 -
        2 * a^2 * chart_y^2 * z

    resultant = a^2 * e - a * b * d + b^2 * c
    slice_equation = a * d + b * c
    inverse_y = 2 * b * d - a * e
    is_constant(resultant, 1) ||
        error("pipeline chart failed resultant normalization")
    is_constant(slice_equation, 1) ||
        error("pipeline chart failed affine slice")
    inverse_y == chart_y ||
        error("pipeline chart failed first inverse coordinate")

    induced = (a * c, a * e + b * d, b * e)
    source_change = (XVAR, YVAR, -rat(1, 2) * TVAR)
    transformed = compose_map(induced, source_change)
    compact = (transformed[3], 2 * transformed[2], 2 * transformed[1])

    u = 1 + XVAR * YVAR
    expected = (
        u^3 * TVAR + YVAR^2 * u * (4 + 3 * XVAR * YVAR),
        YVAR +
            3 * XVAR * u^2 * TVAR +
            3 * XVAR * YVAR^2 * (4 + 3 * XVAR * YVAR),
        2 * XVAR - 3 * XVAR^2 * YVAR - XVAR^3 * TVAR,
    )
    compact == expected ||
        error("pipeline derivation does not match compact certificate")
    is_constant(jacobian_determinant(compact), -2) ||
        error("base pipeline map does not have determinant -2")
    return compact
end

const BASE_POINTS = (
    (rat(0), rat(0), rat(-1, 4)),
    (rat(1), rat(-3, 2), rat(13, 2)),
    (rat(-1), rat(3, 2), rat(13, 2)),
)
const BASE_IMAGE = (rat(-1, 4), rat(0), rat(0))


# ---------------------------------------------------------------------------
# Elementary polynomial automorphisms
# ---------------------------------------------------------------------------

struct Shear
    axis::Int  # zero-based in the shared manifest
    polynomial::Poly
end

function validate_shear(shear::Shear)
    0 <= shear.axis <= 2 || error("invalid shear axis: $(shear.axis)")
    isempty(shear.polynomial.terms) && error("zero shear is not allowed")
    depends_on(shear.polynomial, shear.axis + 1) &&
        error("shear polynomial depends on its modified coordinate")
    return nothing
end

function elementary_map(shear::Shear)
    validate_shear(shear)
    components = Poly[XVAR, YVAR, TVAR]
    index = shear.axis + 1
    components[index] = components[index] + shear.polynomial
    return (components[1], components[2], components[3])
end

function apply_shear(shear::Shear, point::Point)
    validate_shear(shear)
    values = Rat[point...]
    index = shear.axis + 1
    values[index] += evaluate_poly(shear.polynomial, point)
    return (values[1], values[2], values[3])
end

function apply_inverse_shear(shear::Shear, point::Point)
    validate_shear(shear)
    values = Rat[point...]
    index = shear.axis + 1
    values[index] -= evaluate_poly(shear.polynomial, point)
    return (values[1], values[2], values[3])
end

struct ComplexityLimit <: Exception
    reason::String
    message::String
end

Base.showerror(io::IO, error::ComplexityLimit) = print(io, error.message)

function orbit_map(
    base_map::NTuple{3, Poly},
    source_operations::Vector{Shear},
    target_operations::Vector{Shear};
    term_cap::Union{Nothing, Int} = nothing,
    composition_work_cap::Union{Nothing, Int} = nothing,
)
    current = base_map

    for operation in reverse(source_operations)
        elementary = elementary_map(operation)
        if composition_work_cap !== nothing &&
           composition_work_estimate(
               current,
               elementary;
               stop_after = composition_work_cap,
           ) > composition_work_cap
            throw(ComplexityLimit(
                "composition_work_cap",
                "source composition exceeded the work cap",
            ))
        end
        current = compose_map(current, elementary)
        if term_cap !== nothing &&
           maximum(term_count(component) for component in current) > term_cap
            throw(ComplexityLimit(
                "term_cap",
                "source-precomposed map exceeded the term cap",
            ))
        end
    end

    for operation in target_operations
        elementary = elementary_map(operation)
        if composition_work_cap !== nothing &&
           composition_work_estimate(
               elementary,
               current;
               stop_after = composition_work_cap,
           ) > composition_work_cap
            throw(ComplexityLimit(
                "composition_work_cap",
                "target composition exceeded the work cap",
            ))
        end
        current = compose_map(elementary, current)
        if term_cap !== nothing &&
           maximum(term_count(component) for component in current) > term_cap
            throw(ComplexityLimit(
                "term_cap",
                "target-postcomposed map exceeded the term cap",
            ))
        end
    end
    return current
end

function apply_operations(operations::Vector{Shear}, point::Point)
    current = point
    for operation in operations
        current = apply_shear(operation, current)
    end
    return current
end

function apply_inverse_operations(operations::Vector{Shear}, point::Point)
    current = point
    for operation in reverse(operations)
        current = apply_inverse_shear(operation, current)
    end
    return current
end

function monomial_pool(axis_zero::Int, max_degree::Int)
    pool = Exponent[]
    for ex in 0:max_degree, ey in 0:max_degree, et in 0:max_degree
        exponent = (ex, ey, et)
        degree = sum(exponent)
        exponent[axis_zero + 1] == 0 &&
            1 <= degree <= max_degree &&
            push!(pool, exponent)
    end
    return pool
end

function nonzero_integer(rng::AbstractRNG, bound::Int)
    value = 0
    while value == 0
        value = rand(rng, -bound:bound)
    end
    return value
end

function random_shear(
    rng::AbstractRNG,
    max_degree::Int,
    requested_terms::Int,
    coefficient_bound::Int,
)
    axis = rand(rng, 0:2)
    nonconstant = monomial_pool(axis, max_degree)
    isempty(nonconstant) && error("max shear degree must be at least one")

    selected = Exponent[rand(rng, nonconstant)]
    remaining = Exponent[(0, 0, 0)]
    append!(remaining, exponent for exponent in nonconstant
            if exponent != selected[1])
    additional = min(max(requested_terms - 1, 0), length(remaining))
    if additional > 0
        permutation = randperm(rng, length(remaining))
        append!(selected, remaining[permutation[1:additional]])
    end

    raw = Dict{Exponent, Rat}()
    for exponent in selected
        raw[exponent] = rat(nonzero_integer(rng, coefficient_bound))
    end
    shear = Shear(axis, poly(raw))
    validate_shear(shear)
    return shear
end

function random_shear_sequence(
    rng::AbstractRNG,
    depth::Int,
    max_degree::Int,
    shear_terms::Int,
    coefficient_bound::Int,
)
    return Shear[
        random_shear(
            rng,
            max_degree,
            shear_terms,
            coefficient_bound,
        )
        for _ in 1:depth
    ]
end


# ---------------------------------------------------------------------------
# Dependency-free JSON reader and writer
# ---------------------------------------------------------------------------

mutable struct JSONParser
    bytes::Vector{UInt8}
    position::Int
end

at_end(parser::JSONParser) = parser.position > length(parser.bytes)
peek_byte(parser::JSONParser) =
    at_end(parser) ? UInt8(0) : parser.bytes[parser.position]

function skip_whitespace!(parser::JSONParser)
    while !at_end(parser) && peek_byte(parser) in
          (UInt8(' '), UInt8('\n'), UInt8('\r'), UInt8('\t'))
        parser.position += 1
    end
end

function expect_byte!(parser::JSONParser, expected::UInt8)
    at_end(parser) && error("unexpected end of JSON")
    actual = peek_byte(parser)
    actual == expected ||
        error("expected $(Char(expected)), found $(Char(actual))")
    parser.position += 1
end

function parse_json_string!(parser::JSONParser)
    expect_byte!(parser, UInt8('"'))
    output = IOBuffer()
    while !at_end(parser)
        byte = peek_byte(parser)
        parser.position += 1
        byte == UInt8('"') && return String(take!(output))
        if byte == UInt8('\\')
            at_end(parser) && error("unterminated JSON escape")
            escape = peek_byte(parser)
            parser.position += 1
            escape == UInt8('"') && (write(output, UInt8('"')); continue)
            escape == UInt8('\\') && (write(output, UInt8('\\')); continue)
            escape == UInt8('/') && (write(output, UInt8('/')); continue)
            escape == UInt8('b') && (write(output, UInt8(0x08)); continue)
            escape == UInt8('f') && (write(output, UInt8(0x0c)); continue)
            escape == UInt8('n') && (write(output, UInt8('\n')); continue)
            escape == UInt8('r') && (write(output, UInt8('\r')); continue)
            escape == UInt8('t') && (write(output, UInt8('\t')); continue)
            if escape == UInt8('u')
                parser.position + 3 <= length(parser.bytes) ||
                    error("short Unicode escape")
                hexadecimal = String(
                    parser.bytes[parser.position:parser.position + 3],
                )
                parser.position += 4
                codepoint = parse(Int, hexadecimal; base = 16)
                write(output, codeunits(string(Char(codepoint))))
                continue
            end
            error("unsupported JSON escape")
        end
        byte < 0x20 && error("control character in JSON string")
        write(output, byte)
    end
    error("unterminated JSON string")
end

function parse_json_number!(parser::JSONParser)
    start = parser.position
    allowed = Set(UInt8.(collect("-+0123456789.eE")))
    while !at_end(parser) && peek_byte(parser) in allowed
        parser.position += 1
    end
    text = String(parser.bytes[start:parser.position - 1])
    occursin(r"[.eE]", text) && return parse(Float64, text)
    return parse(BigInt, text)
end

function parse_json_array!(parser::JSONParser)
    expect_byte!(parser, UInt8('['))
    skip_whitespace!(parser)
    values = Any[]
    if peek_byte(parser) == UInt8(']')
        parser.position += 1
        return values
    end
    while true
        push!(values, parse_json_value!(parser))
        skip_whitespace!(parser)
        byte = peek_byte(parser)
        byte == UInt8(']') && (parser.position += 1; return values)
        expect_byte!(parser, UInt8(','))
        skip_whitespace!(parser)
    end
end

function parse_json_object!(parser::JSONParser)
    expect_byte!(parser, UInt8('{'))
    skip_whitespace!(parser)
    values = Dict{String, Any}()
    if peek_byte(parser) == UInt8('}')
        parser.position += 1
        return values
    end
    while true
        key = parse_json_string!(parser)
        skip_whitespace!(parser)
        expect_byte!(parser, UInt8(':'))
        skip_whitespace!(parser)
        values[key] = parse_json_value!(parser)
        skip_whitespace!(parser)
        byte = peek_byte(parser)
        byte == UInt8('}') && (parser.position += 1; return values)
        expect_byte!(parser, UInt8(','))
        skip_whitespace!(parser)
    end
end

function consume_literal!(parser::JSONParser, literal::String, value)
    bytes = collect(codeunits(literal))
    stop = parser.position + length(bytes) - 1
    stop <= length(parser.bytes) || error("short JSON literal")
    parser.bytes[parser.position:stop] == bytes ||
        error("invalid JSON literal")
    parser.position = stop + 1
    return value
end

function parse_json_value!(parser::JSONParser)
    skip_whitespace!(parser)
    at_end(parser) && error("unexpected end of JSON")
    byte = peek_byte(parser)
    byte == UInt8('{') && return parse_json_object!(parser)
    byte == UInt8('[') && return parse_json_array!(parser)
    byte == UInt8('"') && return parse_json_string!(parser)
    byte == UInt8('t') && return consume_literal!(parser, "true", true)
    byte == UInt8('f') && return consume_literal!(parser, "false", false)
    byte == UInt8('n') && return consume_literal!(parser, "null", nothing)
    byte in UInt8.(collect("-0123456789")) &&
        return parse_json_number!(parser)
    error("unexpected JSON byte: $(Char(byte))")
end

function parse_json(text::String)
    parser = JSONParser(collect(codeunits(text)), 1)
    value = parse_json_value!(parser)
    skip_whitespace!(parser)
    at_end(parser) || error("trailing data after JSON value")
    return value
end

function json_escape(text::String)
    output = IOBuffer()
    for character in text
        character == '"' && (print(output, "\\\""); continue)
        character == '\\' && (print(output, "\\\\"); continue)
        character == '\b' && (print(output, "\\b"); continue)
        character == '\f' && (print(output, "\\f"); continue)
        character == '\n' && (print(output, "\\n"); continue)
        character == '\r' && (print(output, "\\r"); continue)
        character == '\t' && (print(output, "\\t"); continue)
        if Int(character) < 0x20
            @printf(output, "\\u%04x", Int(character))
        else
            print(output, character)
        end
    end
    return String(take!(output))
end

function emit_json(io::IO, value; level::Int = 0, indent::Int = 2)
    padding = " "^(level * indent)
    child_padding = " "^((level + 1) * indent)

    value === nothing && return print(io, "null")
    value isa Bool && return print(io, value ? "true" : "false")
    value isa Integer && return print(io, value)
    value isa AbstractFloat && begin
        isfinite(value) || error("nonfinite JSON number")
        return print(io, repr(value))
    end
    value isa AbstractString &&
        return print(io, "\"", json_escape(String(value)), "\"")

    if value isa AbstractDict
        keys_sorted = sort!(String.(collect(keys(value))))
        isempty(keys_sorted) && return print(io, "{}")
        print(io, "{\n")
        for (index, key) in enumerate(keys_sorted)
            print(io, child_padding, "\"", json_escape(key), "\": ")
            emit_json(io, value[key]; level = level + 1, indent = indent)
            index < length(keys_sorted) && print(io, ",")
            print(io, "\n")
        end
        print(io, padding, "}")
        return
    end

    if value isa AbstractVector || value isa Tuple
        items = collect(value)
        isempty(items) && return print(io, "[]")
        print(io, "[\n")
        for (index, item) in enumerate(items)
            print(io, child_padding)
            emit_json(io, item; level = level + 1, indent = indent)
            index < length(items) && print(io, ",")
            print(io, "\n")
        end
        print(io, padding, "]")
        return
    end
    error("unsupported JSON value: $(typeof(value))")
end

function write_json(path::String, value)
    directory = dirname(abspath(path))
    isdir(directory) || mkpath(directory)
    open(path, "w") do io
        emit_json(io, value)
        print(io, "\n")
    end
end


# ---------------------------------------------------------------------------
# Shared manifest serialization
# ---------------------------------------------------------------------------

function poly_to_json(value::Poly)
    return Any[
        Dict{String, Any}(
            "exponents" => Any[exponent...],
            "numerator" => numerator(value.terms[exponent]),
            "denominator" => denominator(value.terms[exponent]),
        )
        for exponent in sort!(collect(keys(value.terms)))
    ]
end

function poly_from_json(data)
    raw = Dict{Exponent, Rat}()
    for term in data
        exponents = term["exponents"]
        length(exponents) == 3 || error("invalid serialized exponent")
        exponent = (
            Int(exponents[1]),
            Int(exponents[2]),
            Int(exponents[3]),
        )
        haskey(raw, exponent) &&
            error("duplicate serialized exponent: $exponent")
        raw[exponent] =
            BigInt(term["numerator"]) // BigInt(term["denominator"])
    end
    return poly(raw)
end

function rational_string(value::Rat)
    value_denominator = denominator(value)
    value_denominator == 1 && return string(numerator(value))
    return "$(numerator(value))/$value_denominator"
end

function point_to_json(point::Point)
    return Any[rational_string(coordinate) for coordinate in point]
end

function point_from_json(data)
    length(data) == 3 || error("serialized point must have three coordinates")
    values = Rat[]
    for coordinate in data
        text = string(coordinate)
        separator = if occursin("//", text)
            "//"  # Accept manifests emitted by ACEG Julia 1.0.0.
        elseif occursin("/", text)
            "/"
        else
            nothing
        end
        if separator !== nothing
            parts = split(text, separator; keepempty = true)
            length(parts) == 2 || error("invalid rational coordinate")
            push!(values, parse(BigInt, parts[1]) // parse(BigInt, parts[2]))
        else
            push!(values, parse(BigInt, text) // BigInt(1))
        end
    end
    return (values[1], values[2], values[3])
end

function shear_to_json(shear::Shear)
    validate_shear(shear)
    return Dict{String, Any}(
        "axis" => shear.axis,
        "axis_name" => ("x", "y", "t")[shear.axis + 1],
        "polynomial" => poly_to_json(shear.polynomial),
        "expanded" => polynomial_string(shear.polynomial),
    )
end

function shear_from_json(data)
    shear = Shear(Int(data["axis"]), poly_from_json(data["polynomial"]))
    validate_shear(shear)
    return shear
end

function serialize_map(polynomial_map::NTuple{3, Poly}; include_expanded::Bool)
    records = Any[]
    for (index, component) in enumerate(polynomial_map)
        record = Dict{String, Any}(
            "name" => ("F1", "F2", "F3")[index],
            "degree" => total_degree(component),
            "term_count" => term_count(component),
            "terms" => poly_to_json(component),
        )
        include_expanded &&
            (record["expanded"] = polynomial_string(component))
        push!(records, record)
    end
    return records
end

function deserialize_map(data)
    length(data) == 3 || error("serialized map must have three coordinates")
    values = Poly[poly_from_json(component["terms"]) for component in data]
    return (values[1], values[2], values[3])
end


# ---------------------------------------------------------------------------
# Generation and verification
# ---------------------------------------------------------------------------

function peak_rss_mib()
    try
        return round(Float64(Sys.maxrss()) / (1024.0^2); digits = 3)
    catch
        return nothing
    end
end

function build_candidate(
    base_map::NTuple{3, Poly},
    source_operations::Vector{Shear},
    target_operations::Vector{Shear};
    term_cap::Int,
    composition_work_cap::Int,
    jacobian_work_cap::Int,
    include_expanded::Bool,
    index::Int,
)
    generated = try
        orbit_map(
            base_map,
            source_operations,
            target_operations;
            term_cap = term_cap,
            composition_work_cap = composition_work_cap,
        )
    catch error_value
        error_value isa ComplexityLimit || rethrow()
        return nothing, error_value.reason
    end

    work_estimate = jacobian_work_estimate(generated)
    work_estimate > jacobian_work_cap &&
        return nothing, "jacobian_work_cap"

    determinant = jacobian_determinant(generated)
    is_constant(determinant, -2) || return nothing, "jacobian"

    transported_points = ntuple(
        point_index -> apply_inverse_operations(
            source_operations,
            BASE_POINTS[point_index],
        ),
        3,
    )
    length(Set(transported_points)) == 3 ||
        return nothing, "witness_distinctness"

    expected_image = apply_operations(target_operations, BASE_IMAGE)
    images = ntuple(
        point_index -> evaluate_map(
            generated,
            transported_points[point_index],
        ),
        3,
    )
    all(image -> image == expected_image, images) ||
        return nothing, "collision"

    canonical_hash = map_hash(generated)
    record = Dict{String, Any}(
        "index" => index,
        "id" => @sprintf("ACEG-%04d-%s", index, canonical_hash[1:12]),
        "map_sha256" => canonical_hash,
        "source_automorphism" =>
            Any[shear_to_json(operation) for operation in source_operations],
        "target_automorphism" =>
            Any[shear_to_json(operation) for operation in target_operations],
        "map" => serialize_map(
            generated;
            include_expanded = include_expanded,
        ),
        "jacobian_work_estimate" => work_estimate,
        "jacobian_determinant" => "-2",
        "collision_preimages" =>
            Any[point_to_json(point) for point in transported_points],
        "collision_image" => point_to_json(expected_image),
        "verified" => true,
    )
    return record, nothing
end

function generate_manifest(settings::Dict{String, Any})
    start = time()
    seed = Int(settings["seed"])
    rng = MersenneTwister(seed)
    base_map = derive_pipeline_map()

    all(evaluate_map(base_map, point) == BASE_IMAGE for point in BASE_POINTS) ||
        error("base collision certificate failed")

    maps = Any[]
    hashes = Set{String}()
    rejected = Dict{String, Any}(
        "duplicate" => 0,
        "term_cap" => 0,
        "composition_work_cap" => 0,
        "jacobian_work_cap" => 0,
        "jacobian" => 0,
        "witness_distinctness" => 0,
        "collision" => 0,
    )

    attempts = 0
    while length(maps) < settings["count"] &&
          attempts < settings["attempt_cap"]
        attempts += 1
        source_operations = random_shear_sequence(
            rng,
            settings["source_depth"],
            settings["max_shear_degree"],
            settings["shear_terms"],
            settings["coefficient_bound"],
        )
        target_operations = random_shear_sequence(
            rng,
            settings["target_depth"],
            settings["max_shear_degree"],
            settings["shear_terms"],
            settings["coefficient_bound"],
        )
        record, reason = build_candidate(
            base_map,
            source_operations,
            target_operations;
            term_cap = settings["term_cap"],
            composition_work_cap = settings["composition_work_cap"],
            jacobian_work_cap = settings["jacobian_work_cap"],
            include_expanded = !settings["compact"],
            index = length(maps),
        )
        if record === nothing
            rejected[reason] = rejected[reason] + 1
            continue
        end
        candidate_hash = record["map_sha256"]
        if candidate_hash in hashes
            rejected["duplicate"] = rejected["duplicate"] + 1
            continue
        end
        push!(hashes, candidate_hash)
        push!(maps, record)
    end

    length(maps) == settings["count"] || error(
        "generated $(length(maps)) of $(settings["count"]) requested maps " *
        "after $attempts attempts; raise --attempt-cap or relax complexity",
    )

    largest_degree = maximum(
        Int(component["degree"])
        for record in maps for component in record["map"]
    )
    largest_terms = maximum(
        Int(component["term_count"])
        for record in maps for component in record["map"]
    )

    return Dict{String, Any}(
        "schema" => ACEG_SCHEMA,
        "generator" => "ACEG - Arbitrary Counterexample Generator (Julia)",
        "version" => ACEG_VERSION,
        "scope" =>
            "Exact counterexamples in the polynomial-automorphism orbit of " *
            "the marked-factor pipeline map; no inequivalence claim.",
        "seed" => seed,
        "settings" => Dict{String, Any}(
            "count" => settings["count"],
            "source_depth" => settings["source_depth"],
            "target_depth" => settings["target_depth"],
            "max_shear_degree" => settings["max_shear_degree"],
            "shear_terms" => settings["shear_terms"],
            "coefficient_bound" => settings["coefficient_bound"],
            "term_cap" => settings["term_cap"],
            "composition_work_cap" => settings["composition_work_cap"],
            "jacobian_work_cap" => settings["jacobian_work_cap"],
            "attempt_cap" => settings["attempt_cap"],
            "expanded_formulas_included" => !settings["compact"],
        ),
        "pipeline" => Dict{String, Any}(
            "base_map_sha256" => map_hash(base_map),
            "base_map" => serialize_map(
                base_map;
                include_expanded = !settings["compact"],
            ),
            "base_jacobian_determinant" => "-2",
            "base_collision_preimages" =>
                Any[point_to_json(point) for point in BASE_POINTS],
            "base_collision_image" => point_to_json(BASE_IMAGE),
        ),
        "summary" => Dict{String, Any}(
            "generated" => length(maps),
            "attempts" => attempts,
            "rejected" => rejected,
            "all_verified" => all(record["verified"] for record in maps),
            "all_hashes_distinct" => length(hashes) == length(maps),
            "largest_coordinate_degree" => largest_degree,
            "largest_coordinate_terms" => largest_terms,
            "elapsed_seconds" => round(time() - start; digits = 6),
            "peak_rss_mib" => peak_rss_mib(),
        ),
        "maps" => maps,
    )
end

function verify_manifest_data(manifest)
    manifest_errors = String[]
    map_results = Any[]
    get(manifest, "schema", nothing) == ACEG_SCHEMA ||
        push!(manifest_errors, "unsupported manifest schema")

    base_map = derive_pipeline_map()
    try
        pipeline = manifest["pipeline"]
        stored_base = deserialize_map(pipeline["base_map"])
        stored_base == base_map ||
            push!(
                manifest_errors,
                "stored base map does not match pipeline derivation",
            )
        pipeline["base_map_sha256"] == map_hash(base_map) ||
            push!(manifest_errors, "stored base map hash is invalid")
    catch error_value
        push!(
            manifest_errors,
            "base pipeline record is invalid: $(sprint(showerror, error_value))",
        )
    end

    records = get(manifest, "maps", Any[])
    for (position, record) in enumerate(records)
        local_errors = String[]
        try
            source_operations = Shear[
                shear_from_json(item)
                for item in record["source_automorphism"]
            ]
            target_operations = Shear[
                shear_from_json(item)
                for item in record["target_automorphism"]
            ]
            stored_map = deserialize_map(record["map"])
            rebuilt_map = orbit_map(
                base_map,
                source_operations,
                target_operations,
            )
            stored_map == rebuilt_map ||
                push!(
                    local_errors,
                    "stored map does not match recorded automorphisms",
                )
            map_hash(stored_map) == record["map_sha256"] ||
                push!(local_errors, "map hash mismatch")

            determinant = jacobian_determinant(stored_map)
            is_constant(determinant, -2) ||
                push!(
                    local_errors,
                    "Jacobian is $(polynomial_string(determinant)), not -2",
                )

            stored_points = Point[
                point_from_json(item)
                for item in record["collision_preimages"]
            ]
            rebuilt_points = Point[
                apply_inverse_operations(source_operations, point)
                for point in BASE_POINTS
            ]
            stored_points == rebuilt_points ||
                push!(
                    local_errors,
                    "collision witnesses were not transported correctly",
                )
            length(Set(stored_points)) == 3 ||
                push!(local_errors, "collision witnesses are not distinct")

            expected_image = apply_operations(target_operations, BASE_IMAGE)
            point_from_json(record["collision_image"]) == expected_image ||
                push!(local_errors, "stored collision image is incorrect")
            all(
                evaluate_map(stored_map, point) == expected_image
                for point in stored_points
            ) || push!(local_errors, "collision substitution failed")
        catch error_value
            push!(
                local_errors,
                "invalid record: $(sprint(showerror, error_value))",
            )
        end

        push!(
            map_results,
            Dict{String, Any}(
                "position" => position - 1,
                "id" => get(record, "id", "map-$(position - 1)"),
                "passed" => isempty(local_errors),
                "errors" => local_errors,
            ),
        )
    end

    isempty(records) && push!(manifest_errors, "manifest contains no maps")
    any(!result["passed"] for result in map_results) &&
        push!(manifest_errors, "one or more maps failed verification")
    return Dict{String, Any}(
        "passed" => isempty(manifest_errors),
        "manifest_errors" => manifest_errors,
        "maps_checked" => length(map_results),
        "map_results" => map_results,
    )
end


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

function print_help()
    println("""
usage: julia aceg.jl [generate] [options]
       julia aceg.jl verify MANIFEST
       julia aceg.jl selftest [MANIFEST]
       julia aceg.jl base

Generate options:
  --count N                    maps to generate (default: 5)
  --seed N                     reproducible integer seed
  --source-depth N             source shear depth (default: 2)
  --target-depth N             target shear depth (default: 2)
  --max-shear-degree N         maximum shear degree (default: 2)
  --shear-terms N              terms per shear (default: 2)
  --coefficient-bound N        nonzero integer coefficient bound (default: 3)
  --term-cap N                 coordinate term cap (default: 5000)
  --composition-work-cap N     composition preflight cap (default: 5000000)
  --jacobian-work-cap N        determinant preflight cap (default: 10000000)
  --attempt-cap N              candidate attempt cap (default: 100)
  --compact                    omit expanded formulas
  --output PATH                output manifest (default: aceg_manifest.json)
  --quiet                      suppress generation summary
  --help                       show this help
  --version                    show ACEG version
""")
end

function parse_generate_options(arguments::Vector{String})
    settings = Dict{String, Any}(
        "count" => 5,
        "seed" => nothing,
        "source_depth" => 2,
        "target_depth" => 2,
        "max_shear_degree" => 2,
        "shear_terms" => 2,
        "coefficient_bound" => 3,
        "term_cap" => 5000,
        "composition_work_cap" => 5_000_000,
        "jacobian_work_cap" => 10_000_000,
        "attempt_cap" => 100,
        "compact" => false,
        "output" => "aceg_manifest.json",
        "quiet" => false,
    )
    value_options = Dict(
        "--count" => "count",
        "--seed" => "seed",
        "--source-depth" => "source_depth",
        "--target-depth" => "target_depth",
        "--max-shear-degree" => "max_shear_degree",
        "--shear-terms" => "shear_terms",
        "--coefficient-bound" => "coefficient_bound",
        "--term-cap" => "term_cap",
        "--composition-work-cap" => "composition_work_cap",
        "--jacobian-work-cap" => "jacobian_work_cap",
        "--attempt-cap" => "attempt_cap",
        "--output" => "output",
    )

    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        argument == "--compact" &&
            (settings["compact"] = true; index += 1; continue)
        argument == "--quiet" &&
            (settings["quiet"] = true; index += 1; continue)
        argument == "--help" && (print_help(); return nothing)
        haskey(value_options, argument) ||
            error("unknown generate option: $argument")
        index < length(arguments) ||
            error("missing value for $argument")
        key = value_options[argument]
        raw = arguments[index + 1]
        settings[key] =
            key == "output" ? raw : parse(Int, raw)
        index += 2
    end

    settings["seed"] === nothing &&
        (settings["seed"] = Int(rand(RandomDevice(), UInt64) &
                                UInt64(typemax(Int))))

    positive = (
        "count",
        "max_shear_degree",
        "shear_terms",
        "coefficient_bound",
        "term_cap",
        "composition_work_cap",
        "jacobian_work_cap",
        "attempt_cap",
    )
    all(settings[key] > 0 for key in positive) ||
        error("count, degree, terms, bounds, and caps must be positive")
    settings["source_depth"] >= 0 && settings["target_depth"] >= 0 ||
        error("automorphism depths must be nonnegative")
    settings["source_depth"] == 0 &&
        settings["target_depth"] == 0 &&
        settings["count"] > 1 &&
        error("one automorphism depth must be positive when count > 1")
    return settings
end

function print_generation_summary(manifest, output::String)
    summary = manifest["summary"]
    println("ACEG Julia generation complete")
    println("seed: $(manifest["seed"])")
    println("generated: $(summary["generated"])")
    println("all_verified: $(summary["all_verified"])")
    println("all_hashes_distinct: $(summary["all_hashes_distinct"])")
    println("rejected_attempts: $(summary["rejected"])")
    println("largest_coordinate_degree: $(summary["largest_coordinate_degree"])")
    println("largest_coordinate_terms: $(summary["largest_coordinate_terms"])")
    println("elapsed_seconds: $(summary["elapsed_seconds"])")
    println("peak_rss_mib: $(summary["peak_rss_mib"])")
    println("manifest: $(abspath(output))")
end

function command_generate(arguments::Vector{String})
    settings = parse_generate_options(arguments)
    settings === nothing && return 0
    manifest = generate_manifest(settings)
    output = String(settings["output"])
    write_json(output, manifest)
    verification = verify_manifest_data(manifest)
    verification["passed"] ||
        error("post-write manifest verification failed")
    !settings["quiet"] && print_generation_summary(manifest, output)
    return 0
end

function command_verify(arguments::Vector{String})
    length(arguments) == 1 ||
        error("verify requires exactly one manifest path")
    path = arguments[1]
    manifest = parse_json(read(path, String))
    result = verify_manifest_data(manifest)
    println("manifest: $(abspath(path))")
    println("passed: $(result["passed"])")
    println("maps_checked: $(result["maps_checked"])")
    for record in result["map_results"]
        status = record["passed"] ? "PASS" : "FAIL"
        println("$(record["id"]): $status")
        for message in record["errors"]
            println("  - $message")
        end
    end
    for message in result["manifest_errors"]
        println("manifest error: $message")
    end
    return result["passed"] ? 0 : 1
end

function command_base()
    base_map = derive_pipeline_map()
    println("Pipeline-derived base counterexample")
    for (name, component) in zip(("F1", "F2", "F3"), base_map)
        println("$name = $(polynomial_string(component))")
    end
    println(
        "determinant = ",
        polynomial_string(jacobian_determinant(base_map)),
    )
    println("collision preimages:")
    for point in BASE_POINTS
        println("  $(point_to_json(point))")
    end
    println("collision image: $(point_to_json(BASE_IMAGE))")
    return 0
end

function command_selftest(arguments::Vector{String})
    length(arguments) <= 1 ||
        error("selftest accepts at most one manifest path")

    failures = String[]
    base_map = derive_pipeline_map()
    base_hash = map_hash(base_map)
    base_hash == PYTHON_BASE_MAP_SHA256 ||
        push!(
            failures,
            "base hash mismatch: expected $PYTHON_BASE_MAP_SHA256, got $base_hash",
        )

    determinant = jacobian_determinant(base_map)
    is_constant(determinant, -2) ||
        push!(
            failures,
            "base Jacobian mismatch: $(polynomial_string(determinant))",
        )
    all(evaluate_map(base_map, point) == BASE_IMAGE for point in BASE_POINTS) ||
        push!(failures, "base collision certificate failed")
    length(Set(BASE_POINTS)) == 3 ||
        push!(failures, "base collision witnesses are not distinct")
    canonical_point_serialization =
        point_to_json(BASE_IMAGE) == Any["-1/4", "0", "0"]
    canonical_point_serialization ||
        push!(failures, "canonical rational serialization failed")
    legacy_point_parsing =
        point_from_json(Any["-1//4", "0", "0"]) == BASE_IMAGE
    legacy_point_parsing ||
        push!(failures, "legacy Julia rational parsing failed")

    manifest_checked = false
    maps_checked = 0
    if length(arguments) == 1
        path = arguments[1]
        result = verify_manifest_data(parse_json(read(path, String)))
        manifest_checked = true
        maps_checked = result["maps_checked"]
        if !result["passed"]
            append!(failures, String.(result["manifest_errors"]))
            for record in result["map_results"]
                record["passed"] && continue
                append!(
                    failures,
                    [
                        "$(record["id"]): $message"
                        for message in record["errors"]
                    ],
                )
            end
        end
    end

    println("ACEG Julia self-test")
    println("passed: $(isempty(failures))")
    println("base_map_sha256: $base_hash")
    println("python_hash_parity: $(base_hash == PYTHON_BASE_MAP_SHA256)")
    println("base_jacobian: $(polynomial_string(determinant))")
    println("base_collision: $(all(evaluate_map(base_map, point) == BASE_IMAGE for point in BASE_POINTS))")
    println("canonical_rational_serialization: $canonical_point_serialization")
    println("legacy_rational_parsing: $legacy_point_parsing")
    println("manifest_checked: $manifest_checked")
    manifest_checked && println("maps_checked: $maps_checked")
    for message in failures
        println("failure: $message")
    end
    return isempty(failures) ? 0 : 1
end

function normalized_arguments(arguments::Vector{String})
    commands = Set(["generate", "verify", "selftest", "base"])
    isempty(arguments) && return ["generate"]
    first_argument = arguments[1]
    if !(first_argument in commands) &&
       !(first_argument in ("--help", "-h", "--version"))
        return vcat(["generate"], arguments)
    end
    return arguments
end

function main(arguments::Vector{String})
    args = normalized_arguments(arguments)
    isempty(args) && return command_generate(String[])
    command = args[1]
    command in ("--help", "-h") && (print_help(); return 0)
    command == "--version" && (println(ACEG_VERSION); return 0)
    command == "generate" && return command_generate(args[2:end])
    command == "verify" && return command_verify(args[2:end])
    command == "selftest" && return command_selftest(args[2:end])
    command == "base" && begin
        length(args) == 1 || error("base takes no arguments")
        return command_base()
    end
    error("unknown command: $command")
end

try
    exit(main(ARGS))
catch error_value
    println(stderr, "ACEG Julia error: ", sprint(showerror, error_value))
    exit(1)
end
