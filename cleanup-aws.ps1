# ============================
# cleanup-aws.ps1  (FINAL+ECR FIX)
# Safely tears down an ephemeral EKS+ECR+CloudFront deployment
# and removes chargeable leftovers.
# ============================

# ======== CONFIG (edit if needed) ========
$Region              = "us-east-1"
$ClusterName         = "flask-app-cluster"
$AppName             = "flask-app"
$SvcName             = "$AppName-service"

# ECR cleanup mode:
#   $true  = delete ALL images in ALL repos, then delete ALL repos (default)
#   $false = only delete images in $EcrRepo below, then delete that repo
$AggressiveEcrCleanup = $true
$EcrRepo              = "emotionrecognition-app"   # used if $AggressiveEcrCleanup = $false

# Also delete CloudWatch log groups (/aws/eks/<cluster>/...) to avoid tiny charges
$DeleteEksLogGroups  = $true
# ========================================

$ErrorActionPreference = "Stop"
aws configure set region $Region | Out-Null

function Write-JsonNoBom {
  param([string]$Path, $Object)
  $json = $Object | ConvertTo-Json -Depth 100
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $json, $utf8NoBom)
}

Write-Host "==> Cleanup starting in region $Region" -ForegroundColor Cyan

# 0) Detach ECR read-only policy from node role (if present)
try {
  $ng = aws eks list-nodegroups --cluster-name $ClusterName --query 'nodegroups[0]' --output text 2>$null
  if ($ng -and $ng -ne "None") {
    $roleArn = aws eks describe-nodegroup --cluster-name $ClusterName --nodegroup-name $ng --query 'nodegroup.nodeRole' --output text
    if ($roleArn -and $roleArn -ne "None") {
      $roleName = Split-Path $roleArn -Leaf
      $attached = aws iam list-attached-role-policies --role-name $roleName | ConvertFrom-Json
      foreach ($p in $attached.AttachedPolicies) {
        if ($p.PolicyArn -like "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistry*") {
          Write-Host "Detaching $($p.PolicyArn) from $roleName" -ForegroundColor Yellow
          aws iam detach-role-policy --role-name $roleName --policy-arn $p.PolicyArn | Out-Null
        }
      }
    }
  }
} catch { Write-Host "(Couldn’t inspect/detach node role policies; continuing)" -ForegroundColor DarkYellow }

# 1) K8s objects (delete LB + app)
try {
  Write-Host "Updating kubeconfig for $ClusterName ..." -ForegroundColor Yellow
  aws eks update-kubeconfig --region $Region --name $ClusterName | Out-Null

  Write-Host "Deleting Service $SvcName (removes external ELB) ..." -ForegroundColor Yellow
  kubectl delete service $SvcName --ignore-not-found

  Write-Host "Deleting Deployment $AppName ..." -ForegroundColor Yellow
  kubectl delete deployment $AppName --ignore-not-found
} catch { Write-Host "(Skipping Kubernetes cleanup – kubeconfig/cluster may not exist)" -ForegroundColor DarkYellow }

# 2) CloudFront (disable + delete any distributions pointing to *.elb.amazonaws.com)
try {
  Write-Host "Scanning CloudFront distributions for ELB origins ..." -ForegroundColor Yellow
  $d = aws cloudfront list-distributions --output json | ConvertFrom-Json
  if ($d.DistributionList -and $d.DistributionList.Items) {
    foreach ($dist in $d.DistributionList.Items) {
      $hasElb = $false
      foreach ($o in $dist.Origins.Items) {
        if ($o.DomainName -like "*.elb.amazonaws.com") { $hasElb = $true; break }
      }
      if ($hasElb) {
        $id   = $dist.Id
        $name = $dist.DomainName
        Write-Host "Disabling & deleting CloudFront $id ($name) ..." -ForegroundColor Yellow
        $raw  = aws cloudfront get-distribution-config --id $id | ConvertFrom-Json
        $etag = $raw.ETag
        $conf = $raw.DistributionConfig
        $conf.Enabled = $false
        $tmp = Join-Path $env:TEMP "cf-conf-$id.json"
        Write-JsonNoBom -Path $tmp -Object $conf
        aws cloudfront update-distribution --id $id --if-match $etag --distribution-config file://$tmp | Out-Null

        # Wait to be "Deployed" again before delete
        for ($i=0; $i -lt 60; $i++) {
          $st = aws cloudfront get-distribution --id $id --query 'Distribution.Status' --output text
          if ($st -eq "Deployed") { break }
          Start-Sleep -Seconds 10
        }
        $etag2 = aws cloudfront get-distribution-config --id $id --query ETag --output text
        aws cloudfront delete-distribution --id $id --if-match $etag2 | Out-Null
        Write-Host "Deleted CloudFront $id" -ForegroundColor Green
      }
    }
  } else {
    Write-Host "No CloudFront distributions found." -ForegroundColor DarkYellow
  }
} catch { Write-Host "(Skipping CloudFront cleanup – none or insufficient perms)" -ForegroundColor DarkYellow }

# 3) ECR cleanup helpers
function Test-ValidEcrRepoName {
  param([string]$Name)
  return ($Name -match '^(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*[a-z0-9]+(?:[._-][a-z0-9]+)*$')
}

function Remove-AllImages-InRepo {
  param([string]$RepoName)

  if (-not (Test-ValidEcrRepoName $RepoName)) {
    Write-Host "Skipping invalid repo token: '$RepoName'" -ForegroundColor DarkYellow
    return
  }

  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  $totalDeleted = 0
  $next = $null

  do {
    $args = @(
      "list-images","--repository-name",$RepoName,
      "--filter","tagStatus=ANY",
      "--max-results","1000",
      "--output","json"
    )
    if ($next) { $args += @("--next-token", $next) }

    $resp = aws ecr @args
    $jo = $resp | ConvertFrom-Json
    $ids = $jo.imageIds

    if ($ids -and $ids.Count -gt 0) {
      $tmp = Join-Path $env:TEMP "ecr-$RepoName-ids.json"
      [System.IO.File]::WriteAllText($tmp, ($ids | ConvertTo-Json -Depth 10), $utf8NoBom)
      aws ecr batch-delete-image --repository-name $RepoName --image-ids file://$tmp | Out-Null
      Remove-Item $tmp -ErrorAction SilentlyContinue
      $totalDeleted += $ids.Count
      Write-Host "ECR [$RepoName]: deleted $($ids.Count) images (total: $totalDeleted)" -ForegroundColor DarkYellow
    }

    $next = $jo.nextToken
  } while ($next)

  # Safety loop: drain any remnants (race conditions)
  for ($i=0; $i -lt 5; $i++) {
    $remain = aws ecr list-images --repository-name $RepoName --filter tagStatus=ANY --max-results 1000 --output json | ConvertFrom-Json
    if (-not $remain -or -not $remain.imageIds -or $remain.imageIds.Count -eq 0) { break }
    $tmp2 = Join-Path $env:TEMP "ecr-$RepoName-remnant.json"
    [System.IO.File]::WriteAllText($tmp2, ($remain.imageIds | ConvertTo-Json -Depth 10), $utf8NoBom)
    aws ecr batch-delete-image --repository-name $RepoName --image-ids file://$tmp2 | Out-Null
    Remove-Item $tmp2 -ErrorAction SilentlyContinue
    $totalDeleted += $remain.imageIds.Count
    Write-Host "ECR [$RepoName]: deleted remnant $($remain.imageIds.Count) images (total: $totalDeleted)" -ForegroundColor DarkYellow
  }
}

try {
  if ($AggressiveEcrCleanup) {
    Write-Host "Aggressive ECR cleanup: deleting ALL repos in $Region ..." -ForegroundColor Yellow
    $repos = aws ecr describe-repositories --query "repositories[].repositoryName" --output json | ConvertFrom-Json
    if ($repos) {
      foreach ($repo in $repos) {
        try {
          Remove-AllImages-InRepo -RepoName $repo
          if (Test-ValidEcrRepoName $repo) {
            aws ecr delete-repository --repository-name $repo --force | Out-Null
            Write-Host "Deleted ECR repo $repo" -ForegroundColor Green
          }
        } catch {
          Write-Host "Could not delete ECR repo '$repo' (maybe in use): $($_.Exception.Message)" -ForegroundColor Red
        }
      }
    } else {
      Write-Host "No ECR repos found." -ForegroundColor DarkYellow
    }
  } else {
    Write-Host "Targeted ECR cleanup: $EcrRepo ..." -ForegroundColor Yellow
    if (Test-ValidEcrRepoName $EcrRepo) {
      $repoExists = $true
      try { aws ecr describe-repositories --repository-names $EcrRepo | Out-Null } catch { $repoExists = $false }
      if ($repoExists) {
        Remove-AllImages-InRepo -RepoName $EcrRepo
        aws ecr delete-repository --repository-name $EcrRepo --force | Out-Null
        Write-Host "Deleted ECR repo $EcrRepo" -ForegroundColor Green
      } else {
        Write-Host "ECR repo $EcrRepo not found." -ForegroundColor DarkYellow
      }
    } else {
      Write-Host "Configured EcrRepo name '$EcrRepo' is invalid; skipping." -ForegroundColor Red
    }
  }
} catch { Write-Host "(Skipping ECR cleanup – permissions or API error)" -ForegroundColor DarkYellow }

# 4) Delete EKS cluster (wait)
try {
  Write-Host "Deleting EKS cluster $ClusterName (this can take ~10–20m) ..." -ForegroundColor Yellow
  eksctl delete cluster --name $ClusterName --region $Region --wait
  Write-Host "Deleted EKS cluster $ClusterName" -ForegroundColor Green
} catch {
  Write-Host "(EKS delete failed or already gone; continuing)" -ForegroundColor DarkYellow
}

# 5) Delete leftover eksctl CloudFormation stacks
try {
  Write-Host "Checking leftover CloudFormation stacks for $ClusterName ..." -ForegroundColor Yellow
  $names = aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE ROLLBACK_COMPLETE DELETE_FAILED --query "StackSummaries[?contains(StackName, 'eksctl-$ClusterName')].StackName" --output text
  if ($names) {
    foreach ($n in ($names -split '\s+' | Where-Object { $_ })) {
      Write-Host "Deleting CF stack $n ..." -ForegroundColor Yellow
      aws cloudformation delete-stack --stack-name $n
      aws cloudformation wait stack-delete-complete --stack-name $n
    }
  } else {
    Write-Host "No leftover CF stacks for $ClusterName" -ForegroundColor DarkYellow
  }
} catch { Write-Host "(CloudFormation cleanup skipped/none)" -ForegroundColor DarkYellow }

# 6) Sweep leftover LBs (elbv2 + classic) that look k8s-created
try {
  Write-Host "Sweeping ELBv2 (ALB/NLB) with k8s/eksctl tags ..." -ForegroundColor Yellow
  $lbs = aws elbv2 describe-load-balancers --output json | ConvertFrom-Json
  foreach ($lb in ($lbs.LoadBalancers | Where-Object { $_ })) {
    $arn = $lb.LoadBalancerArn
    try {
      $tags = aws elbv2 describe-tags --resource-arns $arn | ConvertFrom-Json
      $isK8s = $false
      foreach ($t in $tags.TagDescriptions[0].Tags) {
        if ($t.Key -match 'kubernetes|k8s|eksctl|elbv2.k8s.aws') { $isK8s = $true; break }
      }
      if ($isK8s) {
        Write-Host "Deleting leftover ELBv2: $($lb.LoadBalancerName)" -ForegroundColor DarkYellow
        aws elbv2 delete-load-balancer --load-balancer-arn $arn | Out-Null
      }
    } catch { }
  }

  Write-Host "Sweeping Classic ELBs with k8s/eksctl tags ..." -ForegroundColor Yellow
  $clbs = aws elb describe-load-balancers --output json | ConvertFrom-Json
  foreach ($clb in ($clbs.LoadBalancerDescriptions | Where-Object { $_ })) {
    $name = $clb.LoadBalancerName
    try {
      $tags = aws elb describe-tags --load-balancer-names $name | ConvertFrom-Json
      $isK8s = $false
      foreach ($t in $tags.TagDescriptions[0].Tags) {
        if ($t.Key -match 'kubernetes|k8s|eksctl|elbv2.k8s.aws') { $isK8s = $true; break }
      }
      if ($isK8s) {
        Write-Host "Deleting leftover Classic ELB: $name" -ForegroundColor DarkYellow
        aws elb delete-load-balancer --load-balancer-name $name | Out-Null
      }
    } catch { }
  }
} catch { Write-Host "(ELB sweeps skipped)" -ForegroundColor DarkYellow }

# 7) Optional: delete CloudWatch log groups for this cluster
if ($DeleteEksLogGroups) {
  try {
    Write-Host "Deleting CloudWatch log groups for EKS ($ClusterName) ..." -ForegroundColor Yellow
    $prefix = "/aws/eks/$ClusterName/"
    $lg = aws logs describe-log-groups --log-group-name-prefix $prefix | ConvertFrom-Json
    foreach ($g in ($lg.logGroups | Where-Object { $_ })) {
      aws logs delete-log-group --log-group-name $g.logGroupName | Out-Null
      Write-Host "Deleted log group $($g.logGroupName)" -ForegroundColor Green
    }
  } catch { Write-Host "(Log group cleanup skipped)" -ForegroundColor DarkYellow }
}

Write-Host "==> Cleanup complete." -ForegroundColor Cyan
