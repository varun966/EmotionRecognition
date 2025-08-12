# ============================
# sanity.ps1
# Verifies nothing chargeable remains from this stack.
# ============================

$Region = "us-east-1"
aws configure set region $Region | Out-Null

Write-Host "==> Sanity checks (region $Region)" -ForegroundColor Cyan

Write-Host "`n[EKS clusters]" -ForegroundColor Yellow
aws eks list-clusters

Write-Host "`n[ECR repos]" -ForegroundColor Yellow
aws ecr describe-repositories --query "repositories[].repositoryName" --output table 2>$null

Write-Host "`n[CloudFront dists with ELB origins]" -ForegroundColor Yellow
$cf = aws cloudfront list-distributions --output json | ConvertFrom-Json
if ($cf.DistributionList -and $cf.DistributionList.Items) {
  $found = $false
  foreach ($d in $cf.DistributionList.Items) {
    foreach ($o in $d.Origins.Items) {
      if ($o.DomainName -like "*.elb.amazonaws.com") {
        $found = $true
        Write-Host "  ID=$($d.Id)  Domain=$($d.DomainName)" -ForegroundColor DarkYellow
        break
      }
    }
  }
  if (-not $found) { Write-Host "  (none)" }
} else { Write-Host "  (none)" }

Write-Host "`n[Classic ELBs]" -ForegroundColor Yellow
aws elb describe-load-balancers --query "LoadBalancerDescriptions[].DNSName" --output table 2>$null

Write-Host "`n[ALB/NLB (ELBv2)]" -ForegroundColor Yellow
aws elbv2 describe-load-balancers --query "LoadBalancers[].DNSName" --output table 2>$null

Write-Host "`n[CloudFormation stacks mentioning eksctl]" -ForegroundColor Yellow
aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE ROLLBACK_COMPLETE DELETE_FAILED `
  --query "StackSummaries[?contains(StackName, 'eksctl-')].[StackName, StackStatus]" --output table

Write-Host "`n==> Expect all lists above to be empty (or only unrelated resources)." -ForegroundColor Cyan
