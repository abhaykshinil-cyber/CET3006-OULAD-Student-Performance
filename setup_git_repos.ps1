# ============================================================
# Git Setup Script — 3 Research Projects
# Run this ONCE from PowerShell to initialise all repos and push to GitHub
# ============================================================
# HOW TO RUN:
#   1. Open PowerShell  (Start > search "PowerShell")
#   2. Run this first:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   3. Then run:        & "C:\Users\abhay\OneDrive\Documents\RESEARCH PAPER-OULAD Paper\setup_git_repos.ps1"
# ============================================================

$GITHUB_USER = "abhaykshinil-cyber"
$GIT_NAME    = "Abhay Kalathil Shinil"
$GIT_EMAIL   = "abhaykshinil@gmail.com"

$projects = @(
    @{
        Path    = "C:\Users\abhay\OneDrive\Documents\RESEARCH PAPER-OULAD Paper"
        Repo    = "CET3006-OULAD-Student-Performance"
        Message = "Initial commit: CET3006 OULAD student performance prediction research"
        Desc    = "Comparative study of Random Forest, XGBoost, TabNet and FT-Transformer on OULAD (CET3006)"
    },
    @{
        Path    = "C:\Users\abhay\OneDrive\Documents\DATA REFINEMENT RESEARCH"
        Repo    = "CET3006-Data-Refinement-Chest-Xray"
        Message = "Initial commit: CET3006 data refinement — uncertainty-aware chest X-ray pneumonia classification"
        Desc    = "Uncertainty-aware data cleaning pipeline for chest X-ray pneumonia detection (CET3006)"
    },
    @{
        Path    = "C:\Users\abhay\OneDrive\Documents\BUILD-Deep Learning"
        Repo    = "CET3013-Deep-Learning-Assignment"
        Message = "Initial commit: CET3013 deep learning — MNIST ablation study and MOT with Transformers"
        Desc    = "MNIST CNN ablation study and multi-object tracking with Transformers (CET3013)"
    }
)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  GitHub Repository Setup — 3 Research Projects" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "  1. git init each project folder"
Write-Host "  2. Create the initial commit"
Write-Host "  3. Create public repos on github.com/$GITHUB_USER"
Write-Host "  4. Push all three repos"
Write-Host ""
Write-Host "You need a GitHub Personal Access Token (PAT)." -ForegroundColor Yellow
Write-Host "Get one at: github.com > Settings > Developer settings >"
Write-Host "            Personal access tokens > Tokens (classic)"
Write-Host "Required scope: repo (full control)"
Write-Host ""
$PAT = Read-Host "Paste your GitHub PAT here (input is hidden)"

if ($PAT -eq "") {
    Write-Host "No token provided. Will initialise git locally only (no push)." -ForegroundColor Yellow
}

foreach ($proj in $projects) {
    Write-Host ""
    Write-Host "------------------------------------------------------------" -ForegroundColor Green
    Write-Host "  $($proj.Repo)" -ForegroundColor Green
    Write-Host "------------------------------------------------------------" -ForegroundColor Green

    if (-not (Test-Path $proj.Path)) {
        Write-Host "  [SKIP] Folder not found: $($proj.Path)" -ForegroundColor Red
        continue
    }

    Set-Location $proj.Path

    # --- git init ---
    if (-not (Test-Path ".git")) {
        Write-Host "  Initialising git..."
        git init -b main
    } else {
        Write-Host "  Git already initialised."
    }

    git config user.name  $GIT_NAME
    git config user.email $GIT_EMAIL

    # --- stage and commit ---
    Write-Host "  Staging all files..."
    git add .
    $nFiles = (git status --short).Count
    Write-Host "  $nFiles file(s) staged."

    $hasCommit = (git log --oneline 2>$null).Count -gt 0
    if (-not $hasCommit) {
        Write-Host "  Committing..."
        git commit -m $proj.Message
        Write-Host "  Initial commit created." -ForegroundColor Green
    } else {
        Write-Host "  Commit already exists, skipping."
    }

    # --- GitHub ---
    if ($PAT -ne "") {
        $headers = @{
            Authorization = "token $PAT"
            Accept        = "application/vnd.github+json"
            "User-Agent"  = "setup-script"
        }

        # Check / create repo
        $repoExists = $false
        try {
            $null = Invoke-RestMethod `
                -Uri     "https://api.github.com/repos/$GITHUB_USER/$($proj.Repo)" `
                -Headers $headers -Method Get -ErrorAction Stop
            $repoExists = $true
            Write-Host "  GitHub repo already exists." -ForegroundColor Yellow
        } catch { }

        if (-not $repoExists) {
            Write-Host "  Creating GitHub repo '$($proj.Repo)'..."
            $body = @{
                name        = $proj.Repo
                description = $proj.Desc
                private     = $false
                auto_init   = $false
            } | ConvertTo-Json

            try {
                $newRepo = Invoke-RestMethod `
                    -Uri         "https://api.github.com/user/repos" `
                    -Headers     $headers -Method Post `
                    -Body        $body -ContentType "application/json" -ErrorAction Stop
                Write-Host "  Created: $($newRepo.html_url)" -ForegroundColor Green
            } catch {
                Write-Host "  ERROR creating repo: $_" -ForegroundColor Red
                continue
            }
        }

        # Push (embed token in URL temporarily, then reset to clean URL)
        $cleanUrl = "https://github.com/$GITHUB_USER/$($proj.Repo).git"
        $authUrl  = "https://$($GITHUB_USER):$($PAT)@github.com/$GITHUB_USER/$($proj.Repo).git"

        $existingRemote = git remote 2>$null
        if ("origin" -in $existingRemote) {
            git remote set-url origin $authUrl
        } else {
            git remote add origin $authUrl
        }

        Write-Host "  Pushing to GitHub..."
        git push -u origin main

        # Remove token from config immediately after push
        git remote set-url origin $cleanUrl
        Write-Host "  Push complete. Remote reset to clean URL." -ForegroundColor Green
    } else {
        Write-Host "  (Skipping push — no token)" -ForegroundColor DarkGray
        Write-Host "  To push later:"
        Write-Host "    cd `"$($proj.Path)`""
        Write-Host "    git remote add origin https://github.com/$GITHUB_USER/$($proj.Repo).git"
        Write-Host "    git push -u origin main"
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  All done!" -ForegroundColor Cyan
Write-Host ""
foreach ($proj in $projects) {
    Write-Host "  https://github.com/$GITHUB_USER/$($proj.Repo)" -ForegroundColor White
}
Write-Host "============================================================" -ForegroundColor Cyan
