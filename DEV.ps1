<#
.SYNOPSIS
  Outil de développement Luciole_agents_V1 (Windows, PowerShell).

.DESCRIPTION
  Une seule commande pour chaque geste courant. Lancer depuis la racine du dépôt.

  .\DEV.ps1 -Action install           Crée l'environnement Python avec uv (installe uv si absent).
  .\DEV.ps1 -Action test              Tests unitaires (sans GPU ni Docker).
  .\DEV.ps1 -Action lint              Vérification du code avec ruff.
  .\DEV.ps1 -Action test-integration  Tests marqués `integration` (Docker + GPU).
  .\DEV.ps1 -Action smoke             Test de fumée du matériel (GPU, modèles, conteneurs).
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("install", "test", "lint", "test-integration", "smoke")]
    [string]$Action
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }

function Ensure-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Step "uv absent : installation"
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    }
    Write-OK ("uv " + (uv --version))
}

switch ($Action) {
    "install" {
        Ensure-Uv
        Write-Step "Installation de Python 3.12 et des dépendances verrouillées (uv.lock)"
        uv python install 3.12
        uv sync --frozen
        Write-OK "Environnement prêt. Lancer ensuite : .\DEV.ps1 -Action test"
    }
    "test" {
        Ensure-Uv
        Write-Step "Tests unitaires"
        uv run pytest
    }
    "lint" {
        Ensure-Uv
        Write-Step "ruff"
        uv run ruff check .
    }
    "test-integration" {
        Ensure-Uv
        Write-Step "Tests d'intégration (Docker + GPU)"
        uv run pytest -m integration
    }
    "smoke" {
        Ensure-Uv
        Write-Step "Test de fumée du matériel"
        uv run python -m luciole.core.hardware
    }
}
