param(
    [Parameter(Mandatory=$false)]
    [string]$RemoteUrl,
    [Parameter(Mandatory=$false)]
    [string]$GitHubUser,
    [Parameter(Mandatory=$false)]
    [string]$RepoName
)

function Normalize-RepoUrl($url) {
    if (-not $url) { return $null }
    if ($url -notmatch '^https?://') {
        $url = "https://$url"
    }
    if ($url -notmatch '\.git$') {
        $url = "$url.git"
    }
    return $url
}

if (-not $RemoteUrl) {
    if (-not $GitHubUser) {
        $GitHubUser = Read-Host 'Enter your GitHub username'
    }
    if (-not $RepoName) {
        $RepoName = Read-Host 'Enter your GitHub repository name'
    }
    if ($GitHubUser -and $RepoName) {
        $RemoteUrl = "https://github.com/$GitHubUser/$RepoName.git"
    }
}

$RemoteUrl = Normalize-RepoUrl $RemoteUrl

if (-not $RemoteUrl) {
    Write-Error 'GitHub repo URL is required to push.'
    exit 1
}

Write-Host "Using remote URL: $RemoteUrl"

if (-not (Test-Path .git)) {
    Write-Host 'Initializing Git repository...'
    git init
}

Write-Host 'Adding files to git...'
& git add .
Write-Host 'Committing changes...'
& git commit -m 'Prepare repository for GitHub deployment' --allow-empty

Write-Host 'Removing existing origin remote (if present)...'
& git remote remove origin 2>$null
Write-Host 'Adding origin remote...'
& git remote add origin $RemoteUrl
Write-Host 'Setting branch to main...'
& git branch -M main
Write-Host 'Pushing to GitHub...'
& git push -u origin main

Write-Host 'Push complete. If you see authentication errors, make sure your Git credentials are configured.'
