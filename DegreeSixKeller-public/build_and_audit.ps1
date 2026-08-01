$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$toolchainSpec = (Get-Content -LiteralPath (Join-Path $projectRoot "lean-toolchain") -Raw).Trim()
if ($toolchainSpec -ne 'leanprover/lean4:v4.32.2') {
    throw "Unexpected Lean toolchain: $toolchainSpec"
}
$lakeShim = (Get-Command lake.exe -ErrorAction Stop).Source
$elanRoot = Split-Path -Parent (Split-Path -Parent $lakeShim)
$pinnedLake = Join-Path $elanRoot "toolchains\leanprover--lean4---v4.32.2\bin\lake.exe"
if (-not (Test-Path -LiteralPath $pinnedLake)) {
    throw "Pinned Lake executable not found: $pinnedLake"
}

# Keep Git's safe-directory override process-local; do not mutate global Git
# configuration while Lake inspects checked-in dependencies.
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = '*'

function Invoke-LeanStep {
    param(
        [string]$Label,
        [scriptblock]$Command
    )

    Write-Host "`n== $Label ==" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

# This audit consumes the checked-in `lean-toolchain` and `lake-manifest.json`.
# Dependency updates and cache downloads are bootstrap operations, not part of
# the reproducible trust verdict.
Invoke-LeanStep "Build formalization" { & $pinnedLake build }

$leanFiles = @(
    Get-Item (Join-Path $projectRoot "DegreeSixKeller.lean")
    Get-ChildItem `
        -Path (Join-Path $projectRoot "DegreeSixKeller") `
        -Recurse `
        -File `
        -Filter *.lean
)
$holes = $leanFiles | Select-String -Pattern '^\s*axiom\b|\b(sorry|admit)\b'
if ($holes) {
    $holes | Format-Table Path, LineNumber, Line -AutoSize
    throw "Trust audit failed: a local axiom or proof hole was found."
}

$axiomAuditPath = Join-Path $projectRoot "DegreeSixKeller\AxiomAudit.lean"
$probePattern = '^\s*#print\s+axioms\s+(\S+)\s*$'
$expectedProbes = @(
    Get-Content -LiteralPath $axiomAuditPath | ForEach-Object {
        if ($_ -match $probePattern) {
            $Matches[1]
        }
    }
)
$distinctExpectedProbes = @($expectedProbes | Sort-Object -Unique)
if ($expectedProbes.Count -ne $distinctExpectedProbes.Count) {
    throw "Trust audit configuration failed: duplicate #print axioms probes were found."
}

Write-Host "`n== Enforce theorem axiom whitelist ==" -ForegroundColor Cyan
$axiomOutputLines = @(
    & $pinnedLake env lean .\DegreeSixKeller\AxiomAudit.lean 2>&1 |
        ForEach-Object { $_.ToString() }
)
$axiomExitCode = $LASTEXITCODE
$axiomOutputLines | ForEach-Object { Write-Host $_ }
if ($axiomExitCode -ne 0) {
    throw "Axiom audit compilation failed with exit code $axiomExitCode"
}

$axiomOutput = $axiomOutputLines -join "`n"
$reportPattern = "'(?<name>[^']+)'\s+(?:(?:does not depend on any axioms)|(?:depends on axioms:\s*\[(?<axioms>.*?)\]))"
$reportMatches = [regex]::Matches(
    $axiomOutput,
    $reportPattern,
    [System.Text.RegularExpressions.RegexOptions]::Singleline
)

$reports = @{}
foreach ($report in $reportMatches) {
    $name = $report.Groups['name'].Value
    if ($reports.ContainsKey($name)) {
        throw "Trust audit failed: duplicate axiom report for $name"
    }

    $axioms = @()
    if ($report.Groups['axioms'].Success) {
        $body = $report.Groups['axioms'].Value
        if (-not [string]::IsNullOrWhiteSpace($body)) {
            $axioms = @(
                $body.Split(',') |
                    ForEach-Object { $_ -replace '\s+', '' } |
                    Where-Object { $_ -ne '' }
            )
        }
    }
    $reports[$name] = $axioms
}

if ($reports.Count -ne $distinctExpectedProbes.Count) {
    throw "Trust audit failed: expected $($distinctExpectedProbes.Count) axiom reports, parsed $($reports.Count)."
}

$missingReports = @($distinctExpectedProbes | Where-Object { -not $reports.ContainsKey($_) })
if ($missingReports.Count -ne 0) {
    throw "Trust audit failed: missing axiom reports: $($missingReports -join ', ')"
}

$unexpectedReports = @($reports.Keys | Where-Object { $_ -notin $distinctExpectedProbes })
if ($unexpectedReports.Count -ne 0) {
    throw "Trust audit failed: unexpected axiom reports: $($unexpectedReports -join ', ')"
}

$allowedAxioms = @('propext', 'Classical.choice', 'Quot.sound')
$unexpectedAxioms = @()
foreach ($name in $distinctExpectedProbes) {
    foreach ($axiom in $reports[$name]) {
        if ($axiom -notin $allowedAxioms) {
            $unexpectedAxioms += "${name}: ${axiom}"
        }
    }
}
if ($unexpectedAxioms.Count -ne 0) {
    throw "Trust audit failed: unexpected axioms: $($unexpectedAxioms -join '; ')"
}

Write-Host "`nAXIOM_WHITELIST_PASSED ($($reports.Count) reports)" -ForegroundColor Green
Write-Host "BUILD_AND_AUDIT_PASSED" -ForegroundColor Green
