param(
  [string]$Region  = "us-east-1",
  # Optional: pass your cluster name to scope some checks more tightly
  [string]$Cluster = ""
)

aws configure set region $Region | Out-Null
Write-Host "==> Sanity checks (region $Region)" -ForegroundColor Cyan

# ---------------- Core things that cost ----------------
Write-Host "`n[EKS clusters]" -ForegroundColor Yellow
aws eks list-clusters

Write-Host "`n[ECR repositories + image counts]" -ForegroundColor Yellow
$repos = aws ecr describe-repositories --query "repositories[].repositoryName" --output text 2>$null
if ($repos) {
  foreach ($r in ($repos -split "\s+" | Where-Object { $_ })) {
    $cnt = aws ecr describe-images --repository-name $r --query "length(imageDetails)" --output text 2>$null
    Write-Host ("  {0}  (images: {1})" -f $r, ($cnt -as [int])) -ForegroundColor DarkYellow
  }
} else { Write-Host "  (none)" }

Write-Host "`n[EC2 instances running]" -ForegroundColor Yellow
aws ec2 describe-instances `
  --filters Name=instance-state-name,Values=running `
  --query "Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Tags[?Key=='alpha.eksctl.io/cluster-name'].Value|[0]]" `
  --output table 2>$null

Write-Host "`n[NAT Gateways (hourly $)]" -ForegroundColor Yellow
aws ec2 describe-nat-gateways --query "NatGateways[?State!='deleted'].[NatGatewayId,State,VpcId]" --output table 2>$null

Write-Host "`n[Elastic IPs not attached (hourly $)]" -ForegroundColor Yellow
aws ec2 describe-addresses --query "Addresses[?AssociationId==null].[AllocationId,PublicIp]" --output table 2>$null

Write-Host "`n[EBS volumes unattached (storage $)]" -ForegroundColor Yellow
aws ec2 describe-volumes --filters Name=status,Values=available `
  --query "Volumes[].[VolumeId,Size,Tags]" --output table 2>$null

Write-Host "`n[Classic ELBs]" -ForegroundColor Yellow
aws elb describe-load-balancers --query "LoadBalancerDescriptions[].DNSName" --output table 2>$null

Write-Host "`n[ALB/NLB (ELBv2)]" -ForegroundColor Yellow
aws elbv2 describe-load-balancers --query "LoadBalancers[].DNSName" --output table 2>$null

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

Write-Host "`n[CloudWatch Logs for EKS]" -ForegroundColor Yellow
if ($Cluster) {
  aws logs describe-log-groups --log-group-name-prefix "/aws/eks/$Cluster" `
    --query "logGroups[].[logGroupName,storedBytes]" --output table 2>$null
} else {
  aws logs describe-log-groups --log-group-name-prefix "/aws/eks/" `
    --query "logGroups[].[logGroupName,storedBytes]" --output table 2>$null
}

# --------------- Helpful leftovers (mostly free, but good to know) ---------------
Write-Host "`n[VPCs tagged to cluster]" -ForegroundColor Yellow
if ($Cluster) {
  aws ec2 describe-vpcs --filters `
    Name=tag:alpha.eksctl.io/cluster-name,Values=$Cluster `
    Name=tag:kubernetes.io/cluster/$Cluster,Values=owned,shared `
    --query "Vpcs[].VpcId" --output table 2>$null
} else {
  aws ec2 describe-vpcs --filters Name=tag-key,Values=alpha.eksctl.io/cluster-name `
    --query "Vpcs[].{VpcId:VpcId,Cluster:Tags[?Key=='alpha.eksctl.io/cluster-name'].Value|[0]}" --output table 2>$null
}

Write-Host "`n[CloudFormation stacks mentioning eksctl]" -ForegroundColor Yellow
aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE ROLLBACK_COMPLETE DELETE_FAILED `
  --query "StackSummaries[?contains(StackName, 'eksctl-')].[StackName, StackStatus]" --output table

Write-Host "`n==> Expect all lists above to be empty (or only unrelated resources)." -ForegroundColor Cyan
