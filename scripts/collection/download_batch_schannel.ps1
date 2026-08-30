param(
    [Parameter(Mandatory = $true)]
    [string]$UrlsFile,
    [ValidateRange(1, 1000)]
    [int]$Limit = 300,
    [string]$OutputDir = "data/raw",
    [string]$Manifest = "data/raw/collection_manifest.csv"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$urlsPath = (Resolve-Path (Join-Path $projectRoot $UrlsFile)).Path
$outputPath = Join-Path $projectRoot $OutputDir
$manifestPath = Join-Path $projectRoot $Manifest
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

function Read-DotEnvValue {
    param([string]$Name)
    $entry = Get-Content (Join-Path $projectRoot ".env") |
        Where-Object { $_ -match "^$([regex]::Escape($Name))=" } |
        Select-Object -First 1
    if (-not $entry) {
        return ""
    }
    return ($entry -replace "^$([regex]::Escape($Name))=", "").Trim().Trim('"').Trim("'")
}

function Get-ContentId {
    param([string]$Url)
    $match = [regex]::Match($Url, "/(?:posts|videos)/(\d+)", "IgnoreCase")
    if (-not $match.Success) {
        throw "Could not identify a Facebook post/video ID: $Url"
    }
    return $match.Groups[1].Value
}

function Get-CanonicalUrl {
    param([string]$Url)
    $uri = [Uri]$Url.Trim()
    $path = $uri.AbsolutePath.TrimEnd("/") + "/"
    return "https://www.facebook.com$path"
}

function Add-ManifestRow {
    param([pscustomobject]$Row)
    $directory = Split-Path -Parent $manifestPath
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    if (Test-Path $manifestPath) {
        $Row | Export-Csv -Path $manifestPath -NoTypeInformation -Append -Encoding UTF8
    }
    else {
        $Row | Export-Csv -Path $manifestPath -NoTypeInformation -Encoding UTF8
    }
}

function Get-FirstValue {
    param($Item, [string[]]$Names)
    foreach ($name in $Names) {
        $property = $Item.PSObject.Properties[$name]
        if ($property -and -not [string]::IsNullOrWhiteSpace([string]$property.Value)) {
            return [string]$property.Value
        }
    }
    return ""
}

function Test-IsReply {
    param($Item)
    foreach ($name in @("threadingDepth", "depth", "replyDepth")) {
        $property = $Item.PSObject.Properties[$name]
        if ($property -and $null -ne $property.Value) {
            $depth = 0
            if ([int]::TryParse([string]$property.Value, [ref]$depth) -and $depth -gt 0) {
                return $true
            }
        }
    }
    foreach ($name in @("replyToCommentId", "parentCommentId", "parent_comment_id", "replyToId", "parentId", "topCommentId")) {
        $property = $Item.PSObject.Properties[$name]
        if ($property -and -not [string]::IsNullOrWhiteSpace([string]$property.Value) -and [string]$property.Value -ne "0") {
            return $true
        }
    }
    $type = Get-FirstValue $Item @("type")
    $isReply = Get-FirstValue $Item @("isReply")
    return $type.ToLowerInvariant() -eq "reply" -or $isReply.ToLowerInvariant() -eq "true"
}

function Remove-CommentMentions {
    param([string]$Text, $Item)
    $cleaned = [regex]::Replace($Text, "@[\w.-]+", " ")
    $mentionsProperty = $Item.PSObject.Properties["mentions"]
    if ($mentionsProperty) {
        foreach ($mention in @($mentionsProperty.Value)) {
            $name = if ($mention -is [string]) {
                [string]$mention
            }
            else {
                Get-FirstValue $mention @("name", "text", "displayName", "profileName")
            }
            if (-not [string]::IsNullOrWhiteSpace($name)) {
                $escaped = [regex]::Escape($name.Trim())
                $cleaned = [regex]::Replace($cleaned, "(?i)(?<!\w)@?$escaped(?=\s|[:,.!?;-]|$)", " ")
            }
        }
    }
    return ([regex]::Replace($cleaned, "\s+", " ")).Trim(" ", ",", ":", ";", "-")
}

$token = Read-DotEnvValue "APIFY_API_KEY"
if ([string]::IsNullOrWhiteSpace($token)) {
    throw "APIFY_API_KEY is missing from .env"
}
$actor = Read-DotEnvValue "APIFY_ACTOR_ID"
if ([string]::IsNullOrWhiteSpace($actor)) {
    $actor = "apify/facebook-comments-scraper"
}
$actorPath = $actor.Replace("/", "~")
$commentsMode = Read-DotEnvValue "APIFY_COMMENTS_MODE"
$viewOption = switch ($commentsMode.ToUpperInvariant()) {
    "NEWEST" { "RECENT_ACTIVITY" }
    "MOST_RELEVANT" { "RANKED_THREADED" }
    default { "RANKED_UNFILTERED" }
}
$headers = @{
    Authorization = "Bearer $token"
    "Content-Type" = "application/json"
}

$knownIds = @{}
Get-ChildItem $outputPath -Filter "comments_post_*.csv" -ErrorAction SilentlyContinue |
    ForEach-Object {
        $knownIds[$_.BaseName.Replace("comments_post_", "")] = $true
    }
$batchIds = @{}
$urls = Get-Content $urlsPath |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.StartsWith("#") }

$collected = 0
$skipped = 0
$failed = 0

foreach ($rawUrl in $urls) {
    $sourceUrl = Get-CanonicalUrl $rawUrl
    $contentId = Get-ContentId $sourceUrl
    $relativeOutput = "data/raw/comments_post_$contentId.csv"
    $csvPath = Join-Path $projectRoot $relativeOutput
    $collectedAt = [DateTime]::UtcNow.ToString("o")

    if ($knownIds.ContainsKey($contentId)) {
        Write-Output "SKIP existing $contentId"
        Add-ManifestRow ([pscustomobject]@{
            content_id = $contentId
            source_url = $sourceUrl
            requested_limit = $Limit
            output_path = $relativeOutput
            collected_at = $collectedAt
            status = "skipped_existing"
            comment_count = 0
            actor_run_id = ""
            error = ""
        })
        $skipped += 1
        continue
    }
    if ($batchIds.ContainsKey($contentId)) {
        Write-Output "SKIP batch duplicate $contentId"
        $skipped += 1
        continue
    }
    $batchIds[$contentId] = $true

    $runId = ""
    try {
        $actorInput = @{
            startUrls = @(@{ url = $sourceUrl })
            resultsLimit = $Limit
            includeNestedComments = $false
            viewOption = $viewOption
        } | ConvertTo-Json -Depth 5

        Write-Output "START $contentId limit=$Limit"
        $startResponse = Invoke-RestMethod `
            -Method Post `
            -Uri "https://api.apify.com/v2/acts/$actorPath/runs" `
            -Headers $headers `
            -Body $actorInput `
            -TimeoutSec 30
        $run = $startResponse.data
        $runId = [string]$run.id
        if ([string]::IsNullOrWhiteSpace($runId)) {
            throw "Apify did not return a run ID."
        }

        $deadline = [DateTime]::UtcNow.AddMinutes(10)
        $terminal = @("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT")
        while ($terminal -notcontains [string]$run.status) {
            if ([DateTime]::UtcNow -ge $deadline) {
                throw "Actor run $runId exceeded the 10-minute polling deadline."
            }
            Start-Sleep -Seconds 5
            $runResponse = Invoke-RestMethod `
                -Method Get `
                -Uri "https://api.apify.com/v2/actor-runs/$runId" `
                -Headers $headers `
                -TimeoutSec 30
            $run = $runResponse.data
        }
        if ([string]$run.status -ne "SUCCEEDED") {
            throw "Actor run ended with status $($run.status)."
        }

        $datasetId = [string]$run.defaultDatasetId
        if ([string]::IsNullOrWhiteSpace($datasetId)) {
            throw "Successful actor run did not provide a dataset ID."
        }
        $items = Invoke-RestMethod `
            -Method Get `
            -Uri "https://api.apify.com/v2/datasets/$datasetId/items?clean=true&format=json&limit=$Limit" `
            -Headers $headers `
            -TimeoutSec 60

        $rows = @()
        $seenRows = @{}
        $itemIndex = 0
        foreach ($item in @($items)) {
            if ($rows.Count -ge $Limit) {
                break
            }
            if (Test-IsReply $item) {
                continue
            }
            $text = Get-FirstValue $item @("text", "message", "commentText", "body")
            if ([string]::IsNullOrWhiteSpace($text)) {
                continue
            }
            $itemIndex += 1
            $commentId = Get-FirstValue $item @("commentId", "id", "fbId")
            if ([string]::IsNullOrWhiteSpace($commentId)) {
                $commentId = "generated-$contentId-$itemIndex"
            }
            $timestamp = Get-FirstValue $item @("date", "time", "timestamp", "publishedAt", "createdAt", "created_time")
            if ([string]::IsNullOrWhiteSpace($timestamp)) {
                $timestamp = [DateTime]::UtcNow.ToString("o")
            }
            $cleanText = Remove-CommentMentions $text $item
            $dedupeKey = "$commentId`n$cleanText"
            if ($seenRows.ContainsKey($dedupeKey)) {
                continue
            }
            $seenRows[$dedupeKey] = $true
            $rows += [pscustomobject]@{
                comment_id = $commentId
                text = $cleanText
                timestamp = $timestamp
                source_url = $sourceUrl
            }
        }
        $rows = @($rows | Select-Object -First $Limit)
        $rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
        $knownIds[$contentId] = $true
        Add-ManifestRow ([pscustomobject]@{
            content_id = $contentId
            source_url = $sourceUrl
            requested_limit = $Limit
            output_path = $relativeOutput
            collected_at = [DateTime]::UtcNow.ToString("o")
            status = "collected"
            comment_count = $rows.Count
            actor_run_id = $runId
            error = ""
        })
        Write-Output "COLLECTED $contentId count=$($rows.Count)"
        $collected += 1
    }
    catch {
        $message = $_.Exception.Message
        Add-ManifestRow ([pscustomobject]@{
            content_id = $contentId
            source_url = $sourceUrl
            requested_limit = $Limit
            output_path = $relativeOutput
            collected_at = [DateTime]::UtcNow.ToString("o")
            status = "failed"
            comment_count = 0
            actor_run_id = $runId
            error = $message
        })
        Write-Output "FAILED $contentId $message"
        $failed += 1
    }
}

Write-Output "BATCH COMPLETE collected=$collected skipped=$skipped failed=$failed"
if ($failed -gt 0) {
    exit 1
}
