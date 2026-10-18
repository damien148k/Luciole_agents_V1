<#
.SYNOPSIS
  Outil de développement Luciole_agents_V1 (Windows, PowerShell).

.DESCRIPTION
  Une seule commande pour chaque geste courant. Lancer depuis la racine du dépôt.

  .\DEV.ps1 -Action install           Crée l'environnement Python avec uv (installe uv si absent).
  .\DEV.ps1 -Action test              Tests unitaires (sans GPU ni Docker).
  .\DEV.ps1 -Action lint              Vérification du code avec ruff.
  .\DEV.ps1 -Action test-integration  Démarre Qdrant et OpenSearch (Docker), puis les tests `integration`.
  .\DEV.ps1 -Action services-stop     Arrête Qdrant et OpenSearch (les données restent).
  .\DEV.ps1 -Action models            Récupère une fois les modèles (révision épinglée) puis les vérifie.
  .\DEV.ps1 -Action index -Root <dossier> -UseCase ventes
                                      Indexe une racine de dossiers (Qdrant + OpenSearch démarrés).
  .\DEV.ps1 -Action watch -Root <dossier> -UseCase ventes
                                      Surveille la racine et garde l'index à jour (Ctrl+C pour arrêter).
  .\DEV.ps1 -Action smoke             Test de fumée du matériel (GPU, modèles, conteneurs).
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("install", "test", "lint", "test-integration", "services-stop", "models", "index", "watch", "smoke")]
    [string]$Action,
    # Pour -Action index et watch : la racine des dossiers et le cas d'usage.
    [string]$Root,
    [string]$UseCase,
    # Pour -Action index et watch : profil matériel (défaut : celui de cette machine).
    [string]$Hardware
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
        # Extra « embed » : PyTorch et sentence-transformers pour le vrai BGE-M3.
        uv sync --frozen --extra embed
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
        Write-Step "Démarrage de Qdrant et OpenSearch (Docker Desktop doit être lancé)"
        docker compose -f docker/compose.services.yaml up -d --wait
        if ($LASTEXITCODE -ne 0) { throw "Docker : services non démarrés (Docker Desktop est-il lancé ?)" }
        Write-OK "Services prêts"
        Write-Step "Tests d'intégration (Docker + GPU)"
        uv run --extra embed pytest -m integration
    }
    "services-stop" {
        Write-Step "Arrêt de Qdrant et OpenSearch"
        docker compose -f docker/compose.services.yaml down
    }
    "models" {
        Ensure-Uv
        Write-Step "Modèles locaux (configs/models.yaml) : récupération puis vérification"
        uv run python -m luciole.core.model_files fetch
        if ($LASTEXITCODE -ne 0) { throw "Modèles incomplets : voir les messages ci-dessus" }
        Write-OK "Modèles prêts (dossier models\ ou LUCIOLE_MODELS_DIR)"
    }
    "index" {
        Ensure-Uv
        if (-not $Root -or -not $UseCase) { throw "Donner -Root <dossier> et -UseCase <cas d'usage>" }
        Write-Step "Indexation de $Root (cas d'usage $UseCase)"
        $extra = @()
        if ($Hardware) { $extra = @("--hardware", $Hardware) }
        uv run --extra embed python -m luciole.index.pipeline $Root --profile $UseCase @extra
    }
    "watch" {
        Ensure-Uv
        if (-not $Root -or -not $UseCase) { throw "Donner -Root <dossier> et -UseCase <cas d'usage>" }
        Write-Step "Surveillance de $Root (cas d'usage $UseCase), Ctrl+C pour arrêter"
        $extra = @()
        if ($Hardware) { $extra = @("--hardware", $Hardware) }
        uv run --extra embed python -m luciole.index.watcher $Root --profile $UseCase @extra
    }
    "smoke" {
        Ensure-Uv
        Write-Step "Test de fumée du matériel"
        uv run python -m luciole.core.hardware
    }
}
