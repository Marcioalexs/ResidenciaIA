param(
    [string]$Repo = "C:\Users\lacos\Documents\GitHub\ResidenciaIA"
)

$Source = Join-Path $PSScriptRoot "resources"
$Destination = Join-Path $Repo "resources"

Write-Host "Origem:  $Source"
Write-Host "Destino: $Destination"

if (-not (Test-Path $Repo)) {
    throw "Repositorio nao encontrado: $Repo"
}

if (Test-Path $Destination) {
    Write-Host "A pasta resources ja existe. Os arquivos serao mesclados/atualizados."
}

Copy-Item -Path $Source -Destination $Repo -Recurse -Force

Write-Host ""
Write-Host "Resources copiados para o repositorio."
Write-Host "Proximos comandos sugeridos:"
Write-Host "  cd `"$Repo`""
Write-Host "  git status"
Write-Host "  git add resources"
Write-Host '  git commit -m "docs: add C1NC0 research resources"'
Write-Host "  git push origin main"
