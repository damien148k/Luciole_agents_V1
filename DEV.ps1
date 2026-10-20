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
  .\DEV.ps1 -Action search -UseCase ventes -Query "question"
                                      Pose une question à l'index (sens + mots, reranker) et cite les passages.
  .\DEV.ps1 -Action ask -UseCase ventes -Query "question"
                                      Réponse sourcée par le LLM local (Ollama lancé), verdict et trace.
  .\DEV.ps1 -Action agent -UseCase mrae -Query "question"
                                      Réponse par la boucle du cas d'usage (loop.yaml) : chemin, règles, arrêt.
  .\DEV.ps1 -Action bench -UseCase mrae -Reference <fichier.jsonl> [-Answer] [-Agent] [-Runs N]
                                      Banc d'essai sur un jeu de référence : bon tome, bonne page, MRR ;
                                      avec -Answer, le LLM répond aussi (esquives, verdicts) ; avec -Agent,
                                      la réponse passe par la boucle. Deux passages (-Runs pour en changer).
  .\DEV.ps1 -Action api               Lance l'API sur http://127.0.0.1:8000 (POST /api/rag/query, /api/agent/query).
  .\DEV.ps1 -Action smoke             Test de fumée dans l'image Docker avec le GPU (machine, modèles,
                                      PyTorch, embedding, Qdrant et OpenSearch). Construit l'image.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("install", "test", "lint", "test-integration", "services-stop", "models", "index", "watch", "search", "ask", "agent", "bench", "api", "smoke")]
    [string]$Action,
    # Pour -Action index et watch : la racine des dossiers et le cas d'usage.
    [string]$Root,
    [string]$UseCase,
    # Pour -Action search et ask : la question posée.
    [string]$Query,
    # Pour -Action bench : le jeu de référence (hors dépôt), et -Answer pour faire répondre le LLM.
    [string]$Reference,
    [switch]$Answer,
    # Pour -Action bench : répondre par la boucle du cas d'usage plutôt que par le RAG seul.
    [switch]$Agent,
    # Pour -Action bench : nombre de passages (défaut 2).
    [int]$Runs = 0,
    # Pour -Action index, watch, search et smoke : profil matériel (défaut : celui de cette machine).
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
        # -rP : affiche aussi ce que les tests réussis impriment (scores, réponse du LLM).
        uv run --extra embed pytest -m integration -rP
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
    "search" {
        Ensure-Uv
        if (-not $UseCase -or -not $Query) { throw "Donner -UseCase <cas d'usage> et -Query <question>" }
        Write-Step "Recherche dans l'index du cas d'usage $UseCase"
        $extra = @()
        if ($Hardware) { $extra = @("--hardware", $Hardware) }
        uv run --extra embed python -m luciole.retrieval.search $Query --profile $UseCase @extra
        if ($LASTEXITCODE -ne 0) { throw "Recherche en échec : voir le message ci-dessus" }
    }
    "ask" {
        Ensure-Uv
        if (-not $UseCase -or -not $Query) { throw "Donner -UseCase <cas d'usage> et -Query <question>" }
        Write-Step "Réponse sourcée (cas d'usage $UseCase), LLM servi par Ollama"
        $extra = @()
        if ($Hardware) { $extra = @("--hardware", $Hardware) }
        uv run --extra embed python -m luciole.generation.ask $Query --profile $UseCase @extra
        if ($LASTEXITCODE -ne 0) { throw "Réponse en échec : voir le message ci-dessus" }
    }
    "agent" {
        Ensure-Uv
        if (-not $UseCase -or -not $Query) { throw "Donner -UseCase <cas d'usage> et -Query <question>" }
        Write-Step "Réponse par la boucle (cas d'usage $UseCase), LLM servi par Ollama"
        $extra = @()
        if ($Hardware) { $extra = @("--hardware", $Hardware) }
        uv run --extra embed python -m luciole.agents.run $Query --profile $UseCase @extra
        if ($LASTEXITCODE -ne 0) { throw "Boucle en échec : voir le message ci-dessus" }
    }
    "bench" {
        Ensure-Uv
        if (-not $UseCase -or -not $Reference) { throw "Donner -UseCase <cas d'usage> et -Reference <fichier.jsonl>" }
        Write-Step "Banc d'essai (cas d'usage $UseCase) sur $Reference"
        $extra = @()
        if ($Hardware) { $extra += @("--hardware", $Hardware) }
        if ($Answer) { $extra += "--answer" }
        if ($Agent) { $extra += "--agent" }
        if ($Runs -gt 0) { $extra += @("--runs", $Runs) }
        uv run --extra embed python -m luciole.evaluation.bench --profile $UseCase --reference $Reference @extra
        if ($LASTEXITCODE -ne 0) { throw "Banc d'essai en échec : voir le message ci-dessus" }
    }
    "api" {
        Ensure-Uv
        Write-Step "API sur http://127.0.0.1:8000 (Ctrl+C pour arrêter)"
        uv run --extra embed uvicorn luciole.api.app:app --host 127.0.0.1 --port 8000
    }
    "smoke" {
        $compose = @("-f", "docker/compose.services.yaml", "-f", "docker/compose.app.yaml")
        Write-Step "Construction de l'image luciole (la première fois : environ 6 Go, PyTorch CUDA)"
        docker compose @compose build luciole
        if ($LASTEXITCODE -ne 0) { throw "Image non construite (Docker Desktop est-il lancé ?)" }
        Write-Step "Test de fumée dans l'image, avec le GPU"
        $extra = @()
        if ($Hardware) { $extra = @("--hardware", $Hardware) }
        docker compose @compose run --rm luciole python -m luciole.smoke @extra
        if ($LASTEXITCODE -ne 0) { throw "Test de fumée en échec : voir les lignes [ÉCHEC] ci-dessus" }
        Write-OK "Test de fumée vert"
    }
}
