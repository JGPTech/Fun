(*
ACEG - Arbitrary Counterexample Generator (Wolfram Language sister edition)
============================================================================

ACEG derives the compact three-dimensional Jacobian counterexample from the
marked-factor pipeline and generates exactly certified formulas

    G = B o F o A,

where A and B are compositions of elementary determinant-one polynomial
shears. Every generated map uses exact Wolfram Language integers and
rationals. Its full Jacobian determinant is recomputed symbolically, and three
distinct rational collision witnesses are transported through A^(-1) and
checked by direct substitution.

This edition reads and writes the same jgptech.aceg.manifest.v1 schema as the
Python and Julia editions and reproduces their canonical polynomial SHA-256
hashes. It uses only built-in Wolfram Language functionality.

Scope: ACEG generates the polynomial-automorphism orbit of the pipeline map.
It does not claim that its outputs are inequivalent under coordinate changes.
Seeds reproduce Wolfram Language runs; the Python, Julia, and Wolfram random
streams are intentionally language-native and need not select equal orbit
representatives for equal seeds.

Quick start
-----------
    wolframscript -file aceg.wl selftest aceg_manifest.json
    wolframscript -file aceg.wl
    wolframscript -file aceg.wl generate count=5 seed=20260724
    wolframscript -file aceg.wl verify aceg_manifest.json
    wolframscript -file aceg.wl base

If no command is supplied, "generate" is assumed. Exact symbolic expression
growth can be rapid, so conservative term and work caps are applied.
*)

ClearAll["Global`ACEG*"];
ClearAll[x, y, t];

$ACEGVersion = "1.0.1";
$ACEGSchema = "jgptech.aceg.manifest.v1";
$ACEGPythonBaseMapSHA256 =
    "ce70ce88ad5ef1553386ebcfc9ff5b4b1c6d7b239defc514cb66c41bc07423c7";
$ACEGVariables = {x, y, t};
$ACEGErrorTag = "ACEGError";

$ACEGBasePoints = {
    {0, 0, -1/4},
    {1, -3/2, 13/2},
    {-1, 3/2, 13/2}
};
$ACEGBaseImage = {-1/4, 0, 0};


(* ----------------------------------------------------------------------- *)
(* Error handling                                                           *)
(* ----------------------------------------------------------------------- *)

ClearAll[
    ACEGFail,
    ACEGRequire,
    ACEGCapture,
    ACEGCapturedError,
    ACEGGet,
    ACEGKeyExistsQ,
    ACEGDuplicateFreeQ
];

ACEGFail[message_] := Throw[ToString[message], $ACEGErrorTag];

ACEGRequire[condition_, message_] :=
    If[! TrueQ[condition], ACEGFail[message]];

ACEGKeyExistsQ[association_Association, key_] :=
    MemberQ[Keys[association], key];

ACEGGet[association_Association, key_, default_: Missing["NotFound"]] :=
    If[
        ACEGKeyExistsQ[association, key],
        association[key],
        default
    ];

ACEGDuplicateFreeQ[values_List] :=
    Length[values] === Length[DeleteDuplicates[values]];

SetAttributes[ACEGCapture, HoldAll];
ACEGCapture[expression_] :=
    Catch[
        expression,
        $ACEGErrorTag,
        Function[ACEGCapturedError[#1]]
    ];


(* ----------------------------------------------------------------------- *)
(* Exact symbolic polynomial operations                                     *)
(* ----------------------------------------------------------------------- *)

ClearAll[
    ACEGPolynomialTerms,
    ACEGTermCount,
    ACEGTotalDegree,
    ACEGPolynomialZeroQ,
    ACEGPolynomialConstantQ,
    ACEGMapsEqualQ,
    ACEGComposeMap,
    ACEGEvaluateMap,
    ACEGJacobianMatrix,
    ACEGJacobianDeterminant,
    ACEGCompositionWorkEstimate,
    ACEGJacobianWorkEstimate,
    ACEGPolynomialString
];

ACEGPolynomialTerms[polynomial_] := Module[
    {expanded, monomials, rules, exponent, coefficient},
    expanded = Expand[polynomial];
    If[TrueQ[expanded === 0], Return[{}]];
    monomials = If[Head[expanded] === Plus, List @@ expanded, {expanded}];
    rules = Function[monomial,
        exponent = Exponent[monomial, #] & /@ $ACEGVariables;
        coefficient = Together[
            monomial / (
                Times @@ MapThread[Power, {$ACEGVariables, exponent}]
            )
        ];
        exponent -> coefficient
    ] /@ monomials;
    SortBy[rules, First]
];

ACEGTermCount[polynomial_] := Length[ACEGPolynomialTerms[polynomial]];

ACEGTotalDegree[polynomial_] := Module[{terms},
    terms = ACEGPolynomialTerms[polynomial];
    If[terms === {}, -1, Max[Total /@ (First /@ terms)]]
];

ACEGPolynomialZeroQ[polynomial_] := TrueQ[Expand[polynomial] === 0];

ACEGPolynomialConstantQ[polynomial_, expected_] :=
    ACEGPolynomialZeroQ[polynomial - expected];

ACEGMapsEqualQ[left_List, right_List] :=
    Length[left] === 3 &&
    Length[right] === 3 &&
    And @@ MapThread[
        ACEGPolynomialZeroQ[#1 - #2] &,
        {left, right}
    ];

ACEGComposeMap[outer_List, inner_List] := Module[{rules},
    ACEGRequire[
        Length[outer] === 3 && Length[inner] === 3,
        "map composition requires three coordinates"
    ];
    rules = Thread[$ACEGVariables -> inner];
    Expand[# /. rules] & /@ outer
];

ACEGEvaluateMap[polynomialMap_List, point_List] := Module[{rules},
    ACEGRequire[
        Length[polynomialMap] === 3 && Length[point] === 3,
        "map evaluation requires three coordinates"
    ];
    rules = Thread[$ACEGVariables -> point];
    Together[# /. rules] & /@ polynomialMap
];

ACEGJacobianMatrix[polynomialMap_List] :=
    Table[
        D[polynomialMap[[row]], $ACEGVariables[[column]]],
        {row, 1, 3},
        {column, 1, 3}
    ];

ACEGJacobianDeterminant[polynomialMap_List] :=
    Expand[Det[ACEGJacobianMatrix[polynomialMap]]];

ACEGCompositionWorkEstimate[
    outer_List,
    inner_List,
    stopAfter_: None
] := Module[
    {counts, estimate = 0, contribution, exponent},
    counts = Max[ACEGTermCount[#], 1] & /@ inner;
    Do[
        Do[
            exponent = First[termRule];
            contribution = Times @@ MapThread[Power, {counts, exponent}];
            estimate += contribution;
            If[
                IntegerQ[stopAfter] && estimate > stopAfter,
                Throw[estimate]
            ],
            {termRule, ACEGPolynomialTerms[component]}
        ],
        {component, outer}
    ];
    estimate
];

ACEGJacobianWorkEstimate[polynomialMap_List] := Module[
    {counts, a, b, c, d, e, f, g, h, i},
    counts = Table[
        ACEGTermCount[
            D[polynomialMap[[row]], $ACEGVariables[[column]]]
        ],
        {row, 1, 3},
        {column, 1, 3}
    ];
    {{a, b, c}, {d, e, f}, {g, h, i}} = counts;
    a e i + a f h + b d i + b f g + c d h + c e g
];

ACEGPolynomialString[polynomial_] :=
    ToString[Expand[polynomial], InputForm];


(* ----------------------------------------------------------------------- *)
(* Python-compatible canonical polynomial hashes                            *)
(* ----------------------------------------------------------------------- *)

ClearAll[
    ACEGPythonTuple,
    ACEGPythonExponentRepr,
    ACEGPythonPolySignature,
    ACEGMapHash
];

ACEGPythonTuple[items_List] := Switch[
    Length[items],
    0, "()",
    1, "(" <> First[items] <> ",)",
    _, "(" <> StringRiffle[items, ", "] <> ")"
];

ACEGPythonExponentRepr[exponent_List] :=
    "(" <>
    StringRiffle[ToString[#, InputForm] & /@ exponent, ", "] <>
    ")";

ACEGPythonPolySignature[polynomial_] := Module[
    {termStrings},
    termStrings = Function[termRule,
        With[
            {
                exponent = First[termRule],
                coefficient = Last[termRule]
            },
            ACEGPythonTuple[{
                ACEGPythonExponentRepr[exponent],
                ToString[Numerator[coefficient], InputForm],
                ToString[Denominator[coefficient], InputForm]
            }]
        ]
    ] /@ ACEGPolynomialTerms[polynomial];
    ACEGPythonTuple[termStrings]
];

ACEGMapHash[polynomialMap_List] := Module[{canonical},
    canonical = ACEGPythonTuple[
        ACEGPythonPolySignature /@ polynomialMap
    ];
    ToLowerCase[Hash[canonical, "SHA256", "HexString"]]
];


(* ----------------------------------------------------------------------- *)
(* Pipeline derivation and base certificate                                 *)
(* ----------------------------------------------------------------------- *)

ClearAll[ACEGDerivePipelineMap];

ACEGDerivePipelineMap[] := ACEGDerivePipelineMap[] = Module[
    {
        a, chartY, z, b, c, d, e, resultant, sliceEquation,
        inverseY, induced, sourceChange, transformed, compact, u, expected
    },
    {a, chartY, z} = $ACEGVariables;

    b = 1 + a chartY;
    c = 1 - (3/2) a chartY + a^2 z;
    d = (1/2) chartY -
        a z +
        (3/2) a chartY^2 -
        a^2 chartY z;
    e = -2 z +
        4 chartY^2 -
        4 a chartY z +
        3 a chartY^3 -
        2 a^2 chartY^2 z;

    resultant = Expand[a^2 e - a b d + b^2 c];
    sliceEquation = Expand[a d + b c];
    inverseY = Expand[2 b d - a e];

    ACEGRequire[
        ACEGPolynomialConstantQ[resultant, 1],
        "pipeline chart failed resultant normalization"
    ];
    ACEGRequire[
        ACEGPolynomialConstantQ[sliceEquation, 1],
        "pipeline chart failed affine slice"
    ];
    ACEGRequire[
        ACEGPolynomialZeroQ[inverseY - chartY],
        "pipeline chart failed first inverse coordinate"
    ];

    induced = Expand /@ {a c, a e + b d, b e};
    sourceChange = {x, y, -(1/2) t};
    transformed = ACEGComposeMap[induced, sourceChange];
    compact = Expand /@ {
        transformed[[3]],
        2 transformed[[2]],
        2 transformed[[1]]
    };

    u = 1 + x y;
    expected = Expand /@ {
        u^3 t + y^2 u (4 + 3 x y),
        y + 3 x u^2 t + 3 x y^2 (4 + 3 x y),
        2 x - 3 x^2 y - x^3 t
    };

    ACEGRequire[
        ACEGMapsEqualQ[compact, expected],
        "pipeline derivation does not match compact certificate"
    ];
    ACEGRequire[
        ACEGPolynomialConstantQ[
            ACEGJacobianDeterminant[compact],
            -2
        ],
        "base pipeline map does not have determinant -2"
    ];
    compact
];


(* ----------------------------------------------------------------------- *)
(* Elementary polynomial automorphisms                                      *)
(* ----------------------------------------------------------------------- *)

ClearAll[
    ACEGValidateShear,
    ACEGElementaryMap,
    ACEGApplyShear,
    ACEGApplyInverseShear,
    ACEGComplexityFailure,
    ACEGComplexityFailureQ,
    ACEGOrbitMap,
    ACEGApplyOperations,
    ACEGApplyInverseOperations,
    ACEGMonomialPool,
    ACEGNonzeroInteger,
    ACEGMonomialFromExponent,
    ACEGRandomShear,
    ACEGRandomShearSequence
];

ACEGValidateShear[shear_Association] := Module[
    {axis, polynomial},
    axis = ACEGGet[shear, "axis", Missing["axis"]];
    polynomial = ACEGGet[shear, "polynomial", Missing["polynomial"]];
    ACEGRequire[
        IntegerQ[axis] && 0 <= axis <= 2,
        "invalid shear axis"
    ];
    ACEGRequire[
        PolynomialQ[polynomial, $ACEGVariables],
        "invalid shear polynomial"
    ];
    ACEGRequire[
        ! ACEGPolynomialZeroQ[polynomial],
        "zero shear is not allowed"
    ];
    ACEGRequire[
        Exponent[polynomial, $ACEGVariables[[axis + 1]]] === 0,
        "shear polynomial depends on its modified coordinate"
    ];
    Null
];

ACEGElementaryMap[shear_Association] := Module[
    {axis, polynomial, components},
    ACEGValidateShear[shear];
    axis = shear["axis"];
    polynomial = shear["polynomial"];
    components = $ACEGVariables;
    components[[axis + 1]] =
        Expand[components[[axis + 1]] + polynomial];
    components
];

ACEGApplyShear[shear_Association, point_List] := Module[
    {axis, polynomial, updated, value},
    ACEGValidateShear[shear];
    ACEGRequire[Length[point] === 3, "point must have three coordinates"];
    axis = shear["axis"];
    polynomial = shear["polynomial"];
    updated = point;
    value = polynomial /. Thread[$ACEGVariables -> point];
    updated[[axis + 1]] = Together[updated[[axis + 1]] + value];
    updated
];

ACEGApplyInverseShear[shear_Association, point_List] := Module[
    {axis, polynomial, updated, value},
    ACEGValidateShear[shear];
    ACEGRequire[Length[point] === 3, "point must have three coordinates"];
    axis = shear["axis"];
    polynomial = shear["polynomial"];
    updated = point;
    value = polynomial /. Thread[$ACEGVariables -> point];
    updated[[axis + 1]] = Together[updated[[axis + 1]] - value];
    updated
];

ACEGComplexityFailureQ[value_] :=
    MatchQ[value, ACEGComplexityFailure[_, _]];

ACEGOrbitMap[
    baseMap_List,
    sourceOperations_List,
    targetOperations_List,
    termCap_: None,
    compositionWorkCap_: None
] := Module[
    {current, elementary, estimate},
    current = baseMap;

    Do[
        elementary = ACEGElementaryMap[operation];
        If[IntegerQ[compositionWorkCap],
            estimate = ACEGCompositionWorkEstimate[
                current,
                elementary,
                compositionWorkCap
            ];
            If[estimate > compositionWorkCap,
                Throw[ACEGComplexityFailure[
                    "composition_work_cap",
                    "source composition exceeded the work cap"
                ]]
            ]
        ];
        current = ACEGComposeMap[current, elementary];
        If[
            IntegerQ[termCap] &&
            Max[ACEGTermCount /@ current] > termCap,
            Return[ACEGComplexityFailure[
                "term_cap",
                "source-precomposed map exceeded the term cap"
            ]]
        ],
        {operation, Reverse[sourceOperations]}
    ];

    Do[
        elementary = ACEGElementaryMap[operation];
        If[IntegerQ[compositionWorkCap],
            estimate = ACEGCompositionWorkEstimate[
                elementary,
                current,
                compositionWorkCap
            ];
            If[estimate > compositionWorkCap,
                Return[ACEGComplexityFailure[
                    "composition_work_cap",
                    "target composition exceeded the work cap"
                ]]
            ]
        ];
        current = ACEGComposeMap[elementary, current];
        If[
            IntegerQ[termCap] &&
            Max[ACEGTermCount /@ current] > termCap,
            Return[ACEGComplexityFailure[
                "term_cap",
                "target-postcomposed map exceeded the term cap"
            ]]
        ],
        {operation, targetOperations}
    ];
    current
];

ACEGApplyOperations[operations_List, point_List] :=
    Fold[ACEGApplyShear[#2, #1] &, point, operations];

ACEGApplyInverseOperations[operations_List, point_List] :=
    Fold[ACEGApplyInverseShear[#2, #1] &, point, Reverse[operations]];

ACEGMonomialPool[axis_Integer, maxDegree_Integer] :=
    Select[
        Tuples[Range[0, maxDegree], 3],
        #[[axis + 1]] === 0 &&
        1 <= Total[#] <= maxDegree &
    ];

ACEGNonzeroInteger[bound_Integer] := Module[{value = 0},
    While[value === 0, value = RandomInteger[{-bound, bound}]];
    value
];

ACEGMonomialFromExponent[exponent_List] :=
    Times @@ MapThread[Power, {$ACEGVariables, exponent}];

ACEGRandomShear[
    maxDegree_Integer,
    requestedTerms_Integer,
    coefficientBound_Integer
] := Module[
    {
        axis, nonconstant, selected, remaining, additional,
        coefficients, polynomial, shear
    },
    axis = RandomInteger[{0, 2}];
    nonconstant = ACEGMonomialPool[axis, maxDegree];
    ACEGRequire[
        nonconstant =!= {},
        "maximum shear degree must be at least one"
    ];

    selected = {First[RandomSample[nonconstant, 1]]};
    remaining = DeleteCases[
        Prepend[nonconstant, {0, 0, 0}],
        First[selected]
    ];
    additional = Min[Max[requestedTerms - 1, 0], Length[remaining]];
    If[
        additional > 0,
        selected = Join[selected, RandomSample[remaining, additional]]
    ];

    coefficients = ACEGNonzeroInteger[coefficientBound] & /@ selected;
    polynomial = Expand[
        Total[
            MapThread[
                #1 ACEGMonomialFromExponent[#2] &,
                {coefficients, selected}
            ]
        ]
    ];
    shear = <|"axis" -> axis, "polynomial" -> polynomial|>;
    ACEGValidateShear[shear];
    shear
];

ACEGRandomShearSequence[
    depth_Integer,
    maxDegree_Integer,
    shearTerms_Integer,
    coefficientBound_Integer
] :=
    Table[
        ACEGRandomShear[maxDegree, shearTerms, coefficientBound],
        {depth}
    ];


(* ----------------------------------------------------------------------- *)
(* Shared manifest serialization                                            *)
(* ----------------------------------------------------------------------- *)

ClearAll[
    ACEGParseIntegerString,
    ACEGRationalString,
    ACEGParseRationalCoordinate,
    ACEGPointToJSON,
    ACEGPointFromJSON,
    ACEGPolyToJSON,
    ACEGPolyFromJSON,
    ACEGShearToJSON,
    ACEGShearFromJSON,
    ACEGSerializeMap,
    ACEGDeserializeMap,
    ACEGNormalizeJSON,
    ACEGJSONRules,
    ACEGImportJSON,
    ACEGExportJSON
];

ACEGParseIntegerString[text_String] := Module[
    {value},
    ACEGRequire[
        StringMatchQ[text, RegularExpression["[+-]?[0-9]+"]],
        "invalid integer"
    ];
    value = ToExpression[text, InputForm];
    ACEGRequire[IntegerQ[value], "invalid integer"];
    value
];

ACEGRationalString[value_] := Module[{numerator, denominator},
    ACEGRequire[
        IntegerQ[value] || Head[value] === Rational,
        "coordinate is not rational"
    ];
    numerator = Numerator[value];
    denominator = Denominator[value];
    If[
        denominator === 1,
        ToString[numerator, InputForm],
        ToString[numerator, InputForm] <>
        "/" <>
        ToString[denominator, InputForm]
    ]
];

ACEGParseRationalCoordinate[coordinate_] := Module[
    {text, separator, parts, numerator, denominator},
    text = If[
        StringQ[coordinate],
        coordinate,
        ToString[coordinate, InputForm]
    ];
    separator = Which[
        StringContainsQ[text, "//"], "//",
        StringContainsQ[text, "/"], "/",
        True, None
    ];
    If[
        separator === None,
        Return[ACEGParseIntegerString[text]]
    ];
    parts = StringSplit[text, separator];
    ACEGRequire[
        Length[parts] === 2 && AllTrue[parts, StringLength[#] > 0 &],
        "invalid rational coordinate"
    ];
    numerator = ACEGParseIntegerString[parts[[1]]];
    denominator = ACEGParseIntegerString[parts[[2]]];
    ACEGRequire[denominator =!= 0, "zero rational denominator"];
    numerator/denominator
];

ACEGPointToJSON[point_List] := Module[{},
    ACEGRequire[Length[point] === 3, "point must have three coordinates"];
    ACEGRationalString /@ point
];

ACEGPointFromJSON[data_List] := Module[{},
    ACEGRequire[
        Length[data] === 3,
        "serialized point must have three coordinates"
    ];
    ACEGParseRationalCoordinate /@ data
];

ACEGPolyToJSON[polynomial_] :=
    Function[termRule,
        With[
            {
                exponent = First[termRule],
                coefficient = Last[termRule]
            },
            <|
                "exponents" -> exponent,
                "numerator" -> Numerator[coefficient],
                "denominator" -> Denominator[coefficient]
            |>
        ]
    ] /@ ACEGPolynomialTerms[polynomial];

ACEGPolyFromJSON[data_List] := Module[
    {exponents = {}, terms = {}, exponent, numerator, denominator},
    Do[
        ACEGRequire[AssociationQ[term], "invalid polynomial term"];
        exponent = ACEGGet[term, "exponents", Missing["exponents"]];
        numerator = ACEGGet[term, "numerator", Missing["numerator"]];
        denominator = ACEGGet[
            term,
            "denominator",
            Missing["denominator"]
        ];
        ACEGRequire[
            ListQ[exponent] &&
            Length[exponent] === 3 &&
            AllTrue[exponent, IntegerQ[#] && # >= 0 &],
            "invalid serialized exponent"
        ];
        ACEGRequire[
            IntegerQ[numerator] &&
            IntegerQ[denominator] &&
            denominator =!= 0,
            "invalid serialized coefficient"
        ];
        AppendTo[exponents, exponent];
        AppendTo[
            terms,
            (numerator/denominator) ACEGMonomialFromExponent[exponent]
        ],
        {term, data}
    ];
    ACEGRequire[
        ACEGDuplicateFreeQ[exponents],
        "duplicate serialized exponent"
    ];
    Expand[Total[terms]]
];

ACEGShearToJSON[shear_Association] := Module[{axis, polynomial},
    ACEGValidateShear[shear];
    axis = shear["axis"];
    polynomial = shear["polynomial"];
    <|
        "axis" -> axis,
        "axis_name" -> {"x", "y", "t"}[[axis + 1]],
        "polynomial" -> ACEGPolyToJSON[polynomial],
        "expanded" -> ACEGPolynomialString[polynomial]
    |>
];

ACEGShearFromJSON[data_Association] := Module[{shear},
    shear = <|
        "axis" -> ACEGGet[data, "axis", Missing["axis"]],
        "polynomial" -> ACEGPolyFromJSON[
            ACEGGet[data, "polynomial", Missing["polynomial"]]
        ]
    |>;
    ACEGValidateShear[shear];
    shear
];

ACEGSerializeMap[polynomialMap_List, includeExpanded_] := Module[
    {records = {}, record},
    ACEGRequire[
        Length[polynomialMap] === 3,
        "serialized map must have three coordinates"
    ];
    Do[
        record = <|
            "name" -> {"F1", "F2", "F3"}[[index]],
            "degree" -> ACEGTotalDegree[polynomialMap[[index]]],
            "term_count" -> ACEGTermCount[polynomialMap[[index]]],
            "terms" -> ACEGPolyToJSON[polynomialMap[[index]]]
        |>;
        If[
            TrueQ[includeExpanded],
            record = Join[
                record,
                <|
                    "expanded" ->
                        ACEGPolynomialString[polynomialMap[[index]]]
                |>
            ]
        ];
        AppendTo[records, record],
        {index, 1, 3}
    ];
    records
];

ACEGDeserializeMap[data_List] := Module[{values},
    ACEGRequire[
        Length[data] === 3,
        "serialized map must have three coordinates"
    ];
    values = Function[component,
        ACEGRequire[AssociationQ[component], "invalid map component"];
        ACEGPolyFromJSON[
            ACEGGet[component, "terms", Missing["terms"]]
        ]
    ] /@ data;
    values
];

ACEGNormalizeJSON[value_List] := If[
    value =!= {} && AllTrue[value, MatchQ[#, _Rule] &],
    Apply[
        Association,
        Function[rule,
            First[rule] -> ACEGNormalizeJSON[Last[rule]]
        ] /@ value
    ],
    ACEGNormalizeJSON /@ value
];
ACEGNormalizeJSON[value_] := value;

ACEGJSONRules[value_Association] :=
    Normal[Map[ACEGJSONRules, value]];
ACEGJSONRules[value_List] := ACEGJSONRules /@ value;
ACEGJSONRules[value_] := value;

ACEGImportJSON[path_String] := Module[{result},
    result = Quiet[Check[Import[path, "RawJSON"], $Failed]];
    If[
        result === $Failed,
        result = Quiet[Check[Import[path, "JSON"], $Failed]];
        If[result =!= $Failed, result = ACEGNormalizeJSON[result]]
    ];
    ACEGRequire[result =!= $Failed, "could not read JSON file: " <> path];
    result
];

ACEGExportJSON[path_String, data_] := Module[
    {absolute, directory, result},
    absolute = ExpandFileName[path];
    directory = DirectoryName[absolute];
    If[
        ! DirectoryQ[directory],
        Quiet[
            Check[
                CreateDirectory[
                    directory,
                    CreateIntermediateDirectories -> True
                ],
                $Failed
            ]
        ]
    ];
    ACEGRequire[DirectoryQ[directory], "could not create output directory"];
    result = Quiet[Check[Export[absolute, data, "RawJSON"], $Failed]];
    If[
        result === $Failed,
        result = Quiet[
            Check[
                Export[absolute, ACEGJSONRules[data], "JSON"],
                $Failed
            ]
        ]
    ];
    ACEGRequire[result =!= $Failed, "could not write JSON file: " <> path];
    ACEGRequire[
        FileExistsQ[absolute],
        "JSON export returned without creating file: " <> absolute
    ];
    absolute
];


(* ----------------------------------------------------------------------- *)
(* Generation and verification                                              *)
(* ----------------------------------------------------------------------- *)

ClearAll[
    ACEGRejected,
    ACEGRejectedQ,
    ACEGPeakMemoryMiB,
    ACEGBuildCandidate,
    ACEGGenerateManifestSeeded,
    ACEGGenerateManifest,
    ACEGVerifyManifestData
];

ACEGRejectedQ[value_] := MatchQ[value, ACEGRejected[_]];

ACEGPeakMemoryMiB[] := Module[{memory},
    memory = Quiet[Check[MaxMemoryUsed[], $Failed]];
    If[
        NumberQ[memory],
        Round[N[memory/2^20], 0.001],
        Null
    ]
];

ACEGBuildCandidate[
    baseMap_List,
    sourceOperations_List,
    targetOperations_List,
    settings_Association,
    index_Integer
] := Module[
    {
        generated, workEstimate, determinant, transportedPoints,
        expectedImage, images, canonicalHash, record
    },
    generated = ACEGOrbitMap[
        baseMap,
        sourceOperations,
        targetOperations,
        settings["term_cap"],
        settings["composition_work_cap"]
    ];
    If[
        ACEGComplexityFailureQ[generated],
        Return[ACEGRejected[generated[[1]]]]
    ];

    If[
        Max[ACEGTermCount /@ generated] > settings["term_cap"],
        Return[ACEGRejected["term_cap"]]
    ];

    workEstimate = ACEGJacobianWorkEstimate[generated];
    If[
        workEstimate > settings["jacobian_work_cap"],
        Return[ACEGRejected["jacobian_work_cap"]]
    ];

    determinant = ACEGJacobianDeterminant[generated];
    If[
        ! ACEGPolynomialConstantQ[determinant, -2],
        Return[ACEGRejected["jacobian"]]
    ];

    transportedPoints =
        ACEGApplyInverseOperations[sourceOperations, #] & /@
        $ACEGBasePoints;
    If[
        ! ACEGDuplicateFreeQ[transportedPoints],
        Return[ACEGRejected["witness_distinctness"]]
    ];

    expectedImage =
        ACEGApplyOperations[targetOperations, $ACEGBaseImage];
    images = ACEGEvaluateMap[generated, #] & /@ transportedPoints;
    If[
        ! AllTrue[images, SameQ[#, expectedImage] &],
        Return[ACEGRejected["collision"]]
    ];

    canonicalHash = ACEGMapHash[generated];
    record = <|
        "index" -> index,
        "id" -> (
            "ACEG-" <>
            IntegerString[index, 10, 4] <>
            "-" <>
            StringTake[canonicalHash, 12]
        ),
        "map_sha256" -> canonicalHash,
        "source_automorphism" ->
            (ACEGShearToJSON /@ sourceOperations),
        "target_automorphism" ->
            (ACEGShearToJSON /@ targetOperations),
        "map" -> ACEGSerializeMap[
            generated,
            ! TrueQ[settings["compact"]]
        ],
        "jacobian_work_estimate" -> workEstimate,
        "jacobian_determinant" -> "-2",
        "collision_preimages" ->
            (ACEGPointToJSON /@ transportedPoints),
        "collision_image" -> ACEGPointToJSON[expectedImage],
        "verified" -> True
    |>;
    record
];

ACEGGenerateManifestSeeded[settings_Association] := Module[
    {
        start, baseMap, maps = {}, hashes = {}, rejected, attempts = 0,
        sourceOperations, targetOperations, candidate, reason,
        candidateHash, largestDegree, largestTerms, elapsed
    },
    start = AbsoluteTime[];
    baseMap = ACEGDerivePipelineMap[];

    ACEGRequire[
        AllTrue[
            $ACEGBasePoints,
            SameQ[ACEGEvaluateMap[baseMap, #], $ACEGBaseImage] &
        ],
        "base collision certificate failed"
    ];

    rejected = <|
        "duplicate" -> 0,
        "term_cap" -> 0,
        "composition_work_cap" -> 0,
        "jacobian_work_cap" -> 0,
        "jacobian" -> 0,
        "witness_distinctness" -> 0,
        "collision" -> 0
    |>;

    While[
        Length[maps] < settings["count"] &&
        attempts < settings["attempt_cap"],

        attempts++;
        sourceOperations = ACEGRandomShearSequence[
            settings["source_depth"],
            settings["max_shear_degree"],
            settings["shear_terms"],
            settings["coefficient_bound"]
        ];
        targetOperations = ACEGRandomShearSequence[
            settings["target_depth"],
            settings["max_shear_degree"],
            settings["shear_terms"],
            settings["coefficient_bound"]
        ];
        candidate = ACEGBuildCandidate[
            baseMap,
            sourceOperations,
            targetOperations,
            settings,
            Length[maps]
        ];

        If[ACEGRejectedQ[candidate],
            reason = candidate[[1]];
            rejected = Join[
                rejected,
                Association[
                    reason -> (ACEGGet[rejected, reason, 0] + 1)
                ]
            ];
            Continue[]
        ];

        candidateHash = candidate["map_sha256"];
        If[MemberQ[hashes, candidateHash],
            rejected = Join[
                rejected,
                <|"duplicate" -> (rejected["duplicate"] + 1)|>
            ];
            Continue[]
        ];

        AppendTo[hashes, candidateHash];
        AppendTo[maps, candidate];
    ];

    ACEGRequire[
        Length[maps] === settings["count"],
        "generated " <>
        ToString[Length[maps]] <>
        " of " <>
        ToString[settings["count"]] <>
        " requested maps after " <>
        ToString[attempts] <>
        " attempts; raise --attempt-cap or relax complexity"
    ];

    largestDegree = Max[
        Flatten[
            (#["degree"] & /@ #["map"]) & /@ maps
        ]
    ];
    largestTerms = Max[
        Flatten[
            (#["term_count"] & /@ #["map"]) & /@ maps
        ]
    ];
    elapsed = Round[N[AbsoluteTime[] - start], 0.000001];

    <|
        "schema" -> $ACEGSchema,
        "generator" ->
            "ACEG - Arbitrary Counterexample Generator (Wolfram Language)",
        "version" -> $ACEGVersion,
        "scope" ->
            "Exact counterexamples in the polynomial-automorphism orbit of " <>
            "the marked-factor pipeline map; no inequivalence claim.",
        "seed" -> settings["seed"],
        "settings" -> <|
            "count" -> settings["count"],
            "source_depth" -> settings["source_depth"],
            "target_depth" -> settings["target_depth"],
            "max_shear_degree" -> settings["max_shear_degree"],
            "shear_terms" -> settings["shear_terms"],
            "coefficient_bound" -> settings["coefficient_bound"],
            "term_cap" -> settings["term_cap"],
            "composition_work_cap" -> settings["composition_work_cap"],
            "jacobian_work_cap" -> settings["jacobian_work_cap"],
            "attempt_cap" -> settings["attempt_cap"],
            "expanded_formulas_included" ->
                ! TrueQ[settings["compact"]]
        |>,
        "pipeline" -> <|
            "base_map_sha256" -> ACEGMapHash[baseMap],
            "base_map" -> ACEGSerializeMap[
                baseMap,
                ! TrueQ[settings["compact"]]
            ],
            "base_jacobian_determinant" -> "-2",
            "base_collision_preimages" ->
                (ACEGPointToJSON /@ $ACEGBasePoints),
            "base_collision_image" -> ACEGPointToJSON[$ACEGBaseImage]
        |>,
        "summary" -> <|
            "generated" -> Length[maps],
            "attempts" -> attempts,
            "rejected" -> rejected,
            "all_verified" -> AllTrue[maps, TrueQ[#["verified"]] &],
            "all_hashes_distinct" -> ACEGDuplicateFreeQ[hashes],
            "largest_coordinate_degree" -> largestDegree,
            "largest_coordinate_terms" -> largestTerms,
            "elapsed_seconds" -> elapsed,
            "peak_rss_mib" -> ACEGPeakMemoryMiB[]
        |>,
        "maps" -> maps
    |>
];

ACEGGenerateManifest[settings_Association] := Module[{},
    SeedRandom[settings["seed"]];
    ACEGGenerateManifestSeeded[settings]
];

ACEGVerifyManifestData[manifest_Association] := Module[
    {
        manifestErrors = {}, mapResults = {}, baseMap, baseCheck,
        pipeline, storedBase, records, localErrors, recordCheck,
        sourceOperations, targetOperations, storedMap, rebuiltMap,
        determinant, storedPoints, rebuiltPoints, expectedImage,
        storedImage, images, recordID, record
    },
    If[
        ACEGGet[manifest, "schema", Missing["schema"]] =!= $ACEGSchema,
        AppendTo[manifestErrors, "unsupported manifest schema"]
    ];

    baseMap = ACEGDerivePipelineMap[];
    baseCheck = ACEGCapture[
        pipeline = ACEGGet[manifest, "pipeline", Missing["pipeline"]];
        ACEGRequire[AssociationQ[pipeline], "missing pipeline record"];
        storedBase = ACEGDeserializeMap[
            ACEGGet[pipeline, "base_map", Missing["base_map"]]
        ];
        If[
            ! ACEGMapsEqualQ[storedBase, baseMap],
            AppendTo[
                manifestErrors,
                "stored base map does not match pipeline derivation"
            ]
        ];
        If[
            ACEGGet[
                pipeline,
                "base_map_sha256",
                Missing["base_map_sha256"]
            ] =!= ACEGMapHash[baseMap],
            AppendTo[
                manifestErrors,
                "stored base map hash is invalid"
            ]
        ];
        Null
    ];
    If[
        MatchQ[baseCheck, ACEGCapturedError[_]],
        AppendTo[
            manifestErrors,
            "base pipeline record is invalid: " <> ToString[baseCheck[[1]]]
        ]
    ];

    records = ACEGGet[manifest, "maps", {}];
    If[! ListQ[records], records = {}];

    Do[
        record = records[[position]];
        localErrors = {};
        recordID = If[
            AssociationQ[record],
            ToString[
                ACEGGet[
                    record,
                    "id",
                    "map-" <> ToString[position - 1]
                ]
            ],
            "map-" <> ToString[position - 1]
        ];

        recordCheck = ACEGCapture[
            ACEGRequire[AssociationQ[record], "map record is not an object"];
            sourceOperations = ACEGShearFromJSON /@
                ACEGGet[
                    record,
                    "source_automorphism",
                    Missing["source_automorphism"]
                ];
            targetOperations = ACEGShearFromJSON /@
                ACEGGet[
                    record,
                    "target_automorphism",
                    Missing["target_automorphism"]
                ];
            storedMap = ACEGDeserializeMap[
                ACEGGet[record, "map", Missing["map"]]
            ];
            rebuiltMap = ACEGOrbitMap[
                baseMap,
                sourceOperations,
                targetOperations
            ];
            ACEGRequire[
                ! ACEGComplexityFailureQ[rebuiltMap],
                "unexpected rebuilding complexity failure"
            ];

            If[
                ! ACEGMapsEqualQ[storedMap, rebuiltMap],
                AppendTo[
                    localErrors,
                    "stored map does not match recorded automorphisms"
                ]
            ];
            If[
                ACEGMapHash[storedMap] =!=
                    ACEGGet[
                        record,
                        "map_sha256",
                        Missing["map_sha256"]
                    ],
                AppendTo[localErrors, "map hash mismatch"]
            ];

            determinant = ACEGJacobianDeterminant[storedMap];
            If[
                ! ACEGPolynomialConstantQ[determinant, -2],
                AppendTo[
                    localErrors,
                    "Jacobian is " <>
                    ACEGPolynomialString[determinant] <>
                    ", not -2"
                ]
            ];

            storedPoints = ACEGPointFromJSON /@
                ACEGGet[
                    record,
                    "collision_preimages",
                    Missing["collision_preimages"]
                ];
            rebuiltPoints =
                ACEGApplyInverseOperations[sourceOperations, #] & /@
                $ACEGBasePoints;
            If[
                storedPoints =!= rebuiltPoints,
                AppendTo[
                    localErrors,
                    "collision witnesses were not transported correctly"
                ]
            ];
            If[
                ! ACEGDuplicateFreeQ[storedPoints],
                AppendTo[
                    localErrors,
                    "collision witnesses are not distinct"
                ]
            ];

            expectedImage =
                ACEGApplyOperations[targetOperations, $ACEGBaseImage];
            storedImage = ACEGPointFromJSON[
                ACEGGet[
                    record,
                    "collision_image",
                    Missing["collision_image"]
                ]
            ];
            If[
                storedImage =!= expectedImage,
                AppendTo[
                    localErrors,
                    "stored collision image is incorrect"
                ]
            ];

            images = ACEGEvaluateMap[storedMap, #] & /@ storedPoints;
            If[
                ! AllTrue[images, SameQ[#, expectedImage] &],
                AppendTo[
                    localErrors,
                    "collision substitution failed"
                ]
            ];
            Null
        ];

        If[
            MatchQ[recordCheck, ACEGCapturedError[_]],
            AppendTo[
                localErrors,
                "invalid record: " <> ToString[recordCheck[[1]]]
            ]
        ];

        AppendTo[
            mapResults,
            <|
                "position" -> (position - 1),
                "id" -> recordID,
                "passed" -> (localErrors === {}),
                "errors" -> localErrors
            |>
        ],
        {position, 1, Length[records]}
    ];

    If[records === {},
        AppendTo[manifestErrors, "manifest contains no maps"]
    ];
    If[
        AnyTrue[mapResults, ! TrueQ[#["passed"]] &],
        AppendTo[
            manifestErrors,
            "one or more maps failed verification"
        ]
    ];

    <|
        "passed" -> (manifestErrors === {}),
        "manifest_errors" -> manifestErrors,
        "maps_checked" -> Length[mapResults],
        "map_results" -> mapResults
    |>
];


(* ----------------------------------------------------------------------- *)
(* Command-line interface                                                   *)
(* ----------------------------------------------------------------------- *)

ClearAll[
    ACEGBooleanString,
    ACEGPrintHelp,
    ACEGParseGenerateOptions,
    ACEGPrintGenerationSummary,
    ACEGCommandGenerate,
    ACEGCommandVerify,
    ACEGCommandBase,
    ACEGCommandSelftest,
    ACEGExpandGenerateArgument,
    ACEGNormalizedArguments,
    ACEGMainImpl,
    ACEGMain,
    ACEGScriptInvokedQ
];

ACEGBooleanString[value_] := If[TrueQ[value], "true", "false"];

ACEGPrintHelp[] := Print[
"usage: wolframscript -file aceg.wl [generate] [options]\n" <>
"       wolframscript -file aceg.wl verify MANIFEST\n" <>
"       wolframscript -file aceg.wl selftest [MANIFEST]\n" <>
"       wolframscript -file aceg.wl base\n\n" <>
"Generate options:\n" <>
"  --count N                    maps to generate (default: 5)\n" <>
"  --seed N                     reproducible integer seed\n" <>
"  --source-depth N             source shear depth (default: 2)\n" <>
"  --target-depth N             target shear depth (default: 2)\n" <>
"  --max-shear-degree N         maximum shear degree (default: 2)\n" <>
"  --shear-terms N              terms per shear (default: 2)\n" <>
"  --coefficient-bound N        nonzero integer coefficient bound (default: 3)\n" <>
"  --term-cap N                 coordinate term cap (default: 5000)\n" <>
"  --composition-work-cap N     composition preflight cap (default: 5000000)\n" <>
"  --jacobian-work-cap N        determinant preflight cap (default: 10000000)\n" <>
"  --attempt-cap N              candidate attempt cap (default: 100)\n" <>
"  --compact                    omit expanded formulas\n" <>
"  --output PATH                output manifest (default: aceg_manifest.json)\n" <>
"  --quiet                      suppress generation summary\n" <>
"  --help                       show this help\n" <>
"  --version                    show ACEG version\n\n" <>
"Windows-safe syntax (recommended with wolframscript -file):\n" <>
"  generate count=5 seed=20260724 output=aceg_manifest.json"
];

ACEGExpandGenerateArgument[argument_String] := Module[
    {parts, key, value, valueKeys, booleanKeys},
    If[! StringContainsQ[argument, "="], Return[{argument}]];
    parts = StringSplit[argument, "="];
    key = ToLowerCase[
        StringReplace[StringTrim[First[parts]], "_" -> "-"]
    ];
    value = StringTrim[
        StringDrop[argument, StringLength[First[parts]] + 1]
    ];
    valueKeys = {
        "count",
        "seed",
        "source-depth",
        "target-depth",
        "max-shear-degree",
        "shear-terms",
        "coefficient-bound",
        "term-cap",
        "composition-work-cap",
        "jacobian-work-cap",
        "attempt-cap",
        "output"
    };
    booleanKeys = {"compact", "quiet"};
    Which[
        MemberQ[valueKeys, key],
            {"--" <> key, value},
        MemberQ[booleanKeys, key],
            ACEGRequire[
                MemberQ[{"true", "false"}, ToLowerCase[value]],
                key <> " must be true or false"
            ];
            If[ToLowerCase[value] === "true", {"--" <> key}, {}],
        True,
            {argument}
    ]
];

ACEGParseGenerateOptions[arguments_List] := Module[
    {
        settings, valueOptions, normalizedArguments, index = 1,
        argument, key, raw, positiveKeys
    },
    settings = <|
        "count" -> 5,
        "seed" -> Automatic,
        "source_depth" -> 2,
        "target_depth" -> 2,
        "max_shear_degree" -> 2,
        "shear_terms" -> 2,
        "coefficient_bound" -> 3,
        "term_cap" -> 5000,
        "composition_work_cap" -> 5000000,
        "jacobian_work_cap" -> 10000000,
        "attempt_cap" -> 100,
        "compact" -> False,
        "output" -> "aceg_manifest.json",
        "quiet" -> False
    |>;
    valueOptions = <|
        "--count" -> "count",
        "--seed" -> "seed",
        "--source-depth" -> "source_depth",
        "--target-depth" -> "target_depth",
        "--max-shear-degree" -> "max_shear_degree",
        "--shear-terms" -> "shear_terms",
        "--coefficient-bound" -> "coefficient_bound",
        "--term-cap" -> "term_cap",
        "--composition-work-cap" -> "composition_work_cap",
        "--jacobian-work-cap" -> "jacobian_work_cap",
        "--attempt-cap" -> "attempt_cap",
        "--output" -> "output"
    |>;
    normalizedArguments = Flatten[
        ACEGExpandGenerateArgument /@ arguments,
        1
    ];

    While[index <= Length[normalizedArguments],
        argument = normalizedArguments[[index]];
        Which[
            argument === "--compact",
                settings = Join[settings, <|"compact" -> True|>];
                index++,
            argument === "--quiet",
                settings = Join[settings, <|"quiet" -> True|>];
                index++,
            argument === "--help",
                ACEGPrintHelp[];
                Return[ACEGHelpShown],
            ACEGKeyExistsQ[valueOptions, argument],
                ACEGRequire[
                    index < Length[normalizedArguments],
                    "missing value for " <> argument
                ];
                key = valueOptions[argument];
                raw = normalizedArguments[[index + 1]];
                settings = Join[
                    settings,
                    Association[
                        key -> If[
                            key === "output",
                            raw,
                            ACEGParseIntegerString[raw]
                        ]
                    ]
                ];
                index += 2,
            True,
                ACEGFail["unknown generate option: " <> argument]
        ]
    ];

    If[
        settings["seed"] === Automatic,
        settings = Join[
            settings,
            <|"seed" -> RandomInteger[{0, 2^63 - 1}]|>
        ]
    ];

    positiveKeys = {
        "count",
        "max_shear_degree",
        "shear_terms",
        "coefficient_bound",
        "term_cap",
        "composition_work_cap",
        "jacobian_work_cap",
        "attempt_cap"
    };
    ACEGRequire[
        AllTrue[positiveKeys, settings[#] > 0 &],
        "count, degree, terms, bounds, and caps must be positive"
    ];
    ACEGRequire[
        settings["source_depth"] >= 0 &&
        settings["target_depth"] >= 0,
        "automorphism depths must be nonnegative"
    ];
    ACEGRequire[
        ! (
            settings["source_depth"] === 0 &&
            settings["target_depth"] === 0 &&
            settings["count"] > 1
        ),
        "one automorphism depth must be positive when count > 1"
    ];
    settings
];

ACEGPrintGenerationSummary[manifest_Association, output_String] := Module[
    {summary},
    summary = manifest["summary"];
    Print["ACEG Mathematica generation complete"];
    Print["seed: ", manifest["seed"]];
    Print["generated: ", summary["generated"]];
    Print[
        "all_verified: ",
        ACEGBooleanString[summary["all_verified"]]
    ];
    Print[
        "all_hashes_distinct: ",
        ACEGBooleanString[summary["all_hashes_distinct"]]
    ];
    Print[
        "rejected_attempts: ",
        ToString[summary["rejected"], InputForm]
    ];
    Print[
        "largest_coordinate_degree: ",
        summary["largest_coordinate_degree"]
    ];
    Print[
        "largest_coordinate_terms: ",
        summary["largest_coordinate_terms"]
    ];
    Print["elapsed_seconds: ", summary["elapsed_seconds"]];
    Print["peak_rss_mib: ", summary["peak_rss_mib"]];
    Print["manifest: ", ExpandFileName[output]];
];

ACEGCommandGenerate[arguments_List] := Module[
    {settings, manifest, output, verification},
    settings = ACEGParseGenerateOptions[arguments];
    If[settings === ACEGHelpShown, Return[0]];
    manifest = ACEGGenerateManifest[settings];
    output = ACEGExportJSON[settings["output"], manifest];
    verification = ACEGVerifyManifestData[manifest];
    ACEGRequire[
        TrueQ[verification["passed"]],
        "post-write manifest verification failed"
    ];
    If[
        ! TrueQ[settings["quiet"]],
        ACEGPrintGenerationSummary[manifest, output]
    ];
    0
];

ACEGCommandVerify[arguments_List] := Module[
    {path, manifest, result, status},
    ACEGRequire[
        Length[arguments] === 1,
        "verify requires exactly one manifest path"
    ];
    path = First[arguments];
    manifest = ACEGImportJSON[path];
    ACEGRequire[AssociationQ[manifest], "manifest must be a JSON object"];
    result = ACEGVerifyManifestData[manifest];

    Print["manifest: ", ExpandFileName[path]];
    Print["passed: ", ACEGBooleanString[result["passed"]]];
    Print["maps_checked: ", result["maps_checked"]];
    Do[
        status = If[TrueQ[record["passed"]], "PASS", "FAIL"];
        Print[record["id"], ": ", status];
        Scan[Print["  - ", #] &, record["errors"]],
        {record, result["map_results"]}
    ];
    Scan[Print["manifest error: ", #] &, result["manifest_errors"]];
    If[TrueQ[result["passed"]], 0, 1]
];

ACEGCommandBase[] := Module[{baseMap, determinant},
    baseMap = ACEGDerivePipelineMap[];
    determinant = ACEGJacobianDeterminant[baseMap];
    Print["Pipeline-derived base counterexample"];
    Do[
        Print[
            {"F1", "F2", "F3"}[[index]],
            " = ",
            ACEGPolynomialString[baseMap[[index]]]
        ],
        {index, 1, 3}
    ];
    Print["determinant = ", ACEGPolynomialString[determinant]];
    Print["collision preimages:"];
    Scan[Print["  ", ACEGPointToJSON[#]] &, $ACEGBasePoints];
    Print["collision image: ", ACEGPointToJSON[$ACEGBaseImage]];
    0
];

ACEGCommandSelftest[arguments_List] := Module[
    {
        failures = {}, baseMap, baseHash, determinant, baseCollision,
        canonicalSerialization, legacyParsing, sha256UTF8, manifestChecked,
        mapsChecked = 0, manifest, result
    },
    ACEGRequire[
        Length[arguments] <= 1,
        "selftest accepts at most one manifest path"
    ];

    baseMap = ACEGDerivePipelineMap[];
    baseHash = ACEGMapHash[baseMap];
    If[
        baseHash =!= $ACEGPythonBaseMapSHA256,
        AppendTo[
            failures,
            "base hash mismatch: expected " <>
            $ACEGPythonBaseMapSHA256 <>
            ", got " <>
            baseHash
        ]
    ];

    determinant = ACEGJacobianDeterminant[baseMap];
    If[
        ! ACEGPolynomialConstantQ[determinant, -2],
        AppendTo[
            failures,
            "base Jacobian mismatch: " <>
            ACEGPolynomialString[determinant]
        ]
    ];

    baseCollision = AllTrue[
        $ACEGBasePoints,
        SameQ[ACEGEvaluateMap[baseMap, #], $ACEGBaseImage] &
    ];
    If[! baseCollision,
        AppendTo[failures, "base collision certificate failed"]
    ];
    If[! ACEGDuplicateFreeQ[$ACEGBasePoints],
        AppendTo[failures, "base collision witnesses are not distinct"]
    ];

    canonicalSerialization =
        ACEGPointToJSON[$ACEGBaseImage] === {"-1/4", "0", "0"};
    If[! canonicalSerialization,
        AppendTo[failures, "canonical rational serialization failed"]
    ];
    legacyParsing =
        ACEGPointFromJSON[{"-1//4", "0", "0"}] === $ACEGBaseImage;
    If[! legacyParsing,
        AppendTo[failures, "legacy Julia rational parsing failed"]
    ];

    sha256UTF8 =
        ToLowerCase[Hash["abc", "SHA256", "HexString"]] ===
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    If[! sha256UTF8,
        AppendTo[failures, "UTF-8 SHA-256 contract failed"]
    ];

    manifestChecked = Length[arguments] === 1;
    If[manifestChecked,
        manifest = ACEGImportJSON[First[arguments]];
        ACEGRequire[AssociationQ[manifest], "manifest must be a JSON object"];
        result = ACEGVerifyManifestData[manifest];
        mapsChecked = result["maps_checked"];
        If[! TrueQ[result["passed"]],
            failures = Join[failures, result["manifest_errors"]];
            Do[
                If[! TrueQ[record["passed"]],
                    failures = Join[
                        failures,
                        (record["id"] <> ": " <> # & /@ record["errors"])
                    ]
                ],
                {record, result["map_results"]}
            ]
        ]
    ];

    Print["ACEG Mathematica self-test"];
    Print["passed: ", ACEGBooleanString[failures === {}]];
    Print["base_map_sha256: ", baseHash];
    Print[
        "python_hash_parity: ",
        ACEGBooleanString[baseHash === $ACEGPythonBaseMapSHA256]
    ];
    Print["base_jacobian: ", ACEGPolynomialString[determinant]];
    Print["base_collision: ", ACEGBooleanString[baseCollision]];
    Print[
        "canonical_rational_serialization: ",
        ACEGBooleanString[canonicalSerialization]
    ];
    Print[
        "legacy_rational_parsing: ",
        ACEGBooleanString[legacyParsing]
    ];
    Print["utf8_sha256: ", ACEGBooleanString[sha256UTF8]];
    Print[
        "manifest_checked: ",
        ACEGBooleanString[manifestChecked]
    ];
    If[manifestChecked, Print["maps_checked: ", mapsChecked]];
    Scan[Print["failure: ", #] &, failures];
    If[failures === {}, 0, 1]
];

ACEGNormalizedArguments[arguments_List] := Module[
    {commands, normalized, firstArgument},
    commands = {"generate", "verify", "selftest", "base"};
    normalized = arguments;
    While[normalized =!= {} && First[normalized] === "--",
        normalized = Rest[normalized]
    ];
    If[normalized === {}, Return[{"generate"}]];
    firstArgument = First[normalized];
    If[
        ! MemberQ[commands, firstArgument] &&
        ! MemberQ[{"--help", "-h", "--version"}, firstArgument],
        Prepend[normalized, "generate"],
        normalized
    ]
];

ACEGMainImpl[arguments_List] := Module[{args, command},
    args = ACEGNormalizedArguments[arguments];
    command = First[args];
    Switch[command,
        "--help" | "-h",
            ACEGPrintHelp[];
            0,
        "--version",
            Print[$ACEGVersion];
            0,
        "generate",
            ACEGCommandGenerate[Rest[args]],
        "verify",
            ACEGCommandVerify[Rest[args]],
        "selftest",
            ACEGCommandSelftest[Rest[args]],
        "base",
            ACEGRequire[
                Length[args] === 1,
                "base takes no arguments"
            ];
            ACEGCommandBase[],
        _,
            ACEGFail["unknown command: " <> ToString[command]]
    ]
];

ACEGMain[arguments_List] :=
    Catch[
        ACEGMainImpl[arguments],
        $ACEGErrorTag,
        Function[
            Print["ACEG Mathematica error: ", #1];
            1
        ]
    ];

ACEGScriptInvokedQ[] := Module[{commandFile, inputFile},
    If[
        Length[$ScriptCommandLine] < 1 ||
        ! StringQ[$InputFileName] ||
        $InputFileName === "",
        Return[False]
    ];
    commandFile = FileNameTake[First[$ScriptCommandLine]];
    inputFile = FileNameTake[$InputFileName];
    ToLowerCase[commandFile] === ToLowerCase[inputFile]
];

If[
    ACEGScriptInvokedQ[],
    Exit[ACEGMain[Rest[$ScriptCommandLine]]]
];
