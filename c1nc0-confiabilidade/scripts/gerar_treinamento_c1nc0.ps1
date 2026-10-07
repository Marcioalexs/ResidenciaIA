param(
    [Parameter(Mandatory = $false)]
    [string]$Dataset
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($Dataset)) {
    $Dataset = Read-Host "Informe o caminho completo do CSV consolidado"
}

$Dataset = $Dataset.Trim('"')

if (-not (Test-Path -LiteralPath $Dataset)) {
    throw "Dataset nao encontrado: $Dataset"
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

Write-Host ""
Write-Host "C1NC0 - Geracao completa dos artefatos de treinamento" -ForegroundColor Cyan
Write-Host "Projeto : $ProjectRoot"
Write-Host "Dataset : $Dataset"
Write-Host "Python  : $Python"
Write-Host ""

& $Python ".\scripts\gerar_pacote_treinamento_c1nc0.py" --dataset $Dataset

if ($LASTEXITCODE -ne 0) {
    throw "O treinamento terminou com codigo $LASTEXITCODE."
}

Write-Host ""
Write-Host "Concluido. Verifique a pasta dist para o ZIP final." -ForegroundColor Green
