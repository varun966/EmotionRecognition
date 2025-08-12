param(
  [Parameter(Mandatory=$true)][string]$VpcId,
  [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"
aws configure set region $Region | Out-Null

Write-Host "==> Force deleting VPC $VpcId in $Region" -ForegroundColor Cyan

function _ok($msg){ Write-Host "  ✓ $msg" -ForegroundColor Green }
function _skip($msg,$e){ Write-Host "  … $msg skipped: $($e -replace "`n"," ") " -ForegroundColor DarkYellow }

function Remove-NonLocalRoutesAndDelete-RT($rtId){
  # Disassociate all non-main associations
  try {
    $assocIds = aws ec2 describe-route-tables --route-table-ids $rtId `
      --query "RouteTables[0].Associations[?Main!=`true`].RouteTableAssociationId" --output text 2>$null
    if ($assocIds) {
      foreach($a in ($assocIds -split "\s+" | ? {$_})) {
        try { aws ec2 disassociate-route-table --association-id $a 2>$null | Out-Null } catch {}
      }
    }
  } catch {}

  # Delete all non-local routes (IPv4/IPv6)
  try {
    $routes = aws ec2 describe-route-tables --route-table-ids $rtId `
      --query "RouteTables[0].Routes[].{d4:DestinationCidrBlock,d6:DestinationIpv6CidrBlock,origin:Origin}" `
      --output json 2>$null | ConvertFrom-Json
    if ($routes){
      foreach($r in $routes){
        if ($r.origin -ne "CreateRouteTable"){
          if ($r.d4){ try { aws ec2 delete-route --route-table-id $rtId --destination-cidr-block $r.d4 2>$null | Out-Null } catch {} }
          elseif ($r.d6){ try { aws ec2 delete-route --route-table-id $rtId --destination-ipv6-cidr-block $r.d6 2>$null | Out-Null } catch {} }
        }
      }
    }
  } catch {}

  # Retry delete a few times to ride out eventual consistency
  for($i=1;$i -le 6;$i++){
    try {
      aws ec2 delete-route-table --route-table-id $rtId 2>$null | Out-Null
      _ok "Deleted RT $rtId"
      return
    } catch {
      Start-Sleep -Seconds 5
    }
  }
  throw "RT $rtId still has dependencies after retries."
}

# 0) Terminate any instances left in this VPC (defensive)
try {
  $instIds = aws ec2 describe-instances `
    --filters Name=vpc-id,Values=$VpcId Name=instance-state-name,Values=pending,running,stopping,stopped `
    --query "Reservations[].Instances[].InstanceId" --output text 2>$null
  if ($instIds) {
    Write-Host "Terminating instances in VPC: $instIds" -ForegroundColor Yellow
    aws ec2 terminate-instances --instance-ids $instIds 2>$null | Out-Null
    try { aws ec2 wait instance-terminated --instance-ids $instIds 2>$null } catch {}
    _ok "Instances terminated"
  }
} catch { _skip "Terminate instances" $_ }

# 1) ELBv2 (ALB/NLB) + Target Groups
try {
  $lbs = aws elbv2 describe-load-balancers --query "LoadBalancers[?VpcId=='$VpcId'].LoadBalancerArn" --output text 2>$null
  if ($lbs) {
    $tgs = aws elbv2 describe-target-groups --query "TargetGroups[?VpcId=='$VpcId'].TargetGroupArn" --output text 2>$null
    foreach($tg in ($tgs -split "\s+" | ? {$_})) { try { aws elbv2 delete-target-group --target-group-arn $tg 2>$null | Out-Null } catch {} }
    foreach($lb in ($lbs -split "\s+" | ? {$_})) { try { aws elbv2 delete-load-balancer --load-balancer-arn $lb 2>$null | Out-Null } catch {} }
    Start-Sleep 8
    _ok "Deleted ELBv2 + Target Groups"
  }
} catch { _skip "ELBv2 cleanup" $_ }

# 2) Classic ELBs
try {
  $clbs = aws elb describe-load-balancers --query "LoadBalancerDescriptions[?VPCId=='$VpcId'].LoadBalancerName" --output text 2>$null
  foreach($n in ($clbs -split "\s+" | ? {$_})) { try { aws elb delete-load-balancer --load-balancer-name $n 2>$null | Out-Null } catch {} }
  if ($clbs) { _ok "Deleted Classic ELBs" }
} catch { _skip "Classic ELB cleanup" $_ }

# 3) NAT Gateways (wait for deletion)
try {
  $ngs = aws ec2 describe-nat-gateways --filter Name=vpc-id,Values=$VpcId `
    --query "NatGateways[?State!='deleted'].NatGatewayId" --output text 2>$null
  foreach($ng in ($ngs -split "\s+" | ? {$_})) { try { aws ec2 delete-nat-gateway --nat-gateway-id $ng 2>$null | Out-Null } catch {} }
  if ($ngs){
    Write-Host "  … Waiting for NAT Gateways to delete" -ForegroundColor DarkYellow
    for($i=0;$i -lt 60;$i++){
      $left = aws ec2 describe-nat-gateways --filter Name=vpc-id,Values=$VpcId `
        --query "NatGateways[?State!='deleted'].NatGatewayId" --output text 2>$null
      if (-not $left) { break }
      Start-Sleep -Seconds 10
    }
    _ok "NAT Gateways deleted"
  }
} catch { _skip "NAT GW cleanup" $_ }

# 4) VPC Endpoints (Interface/Gateway)
try {
  $eps = aws ec2 describe-vpc-endpoints --filters Name=vpc-id,Values=$VpcId `
    --query "VpcEndpoints[].VpcEndpointId" --output text 2>$null
  if ($eps) { aws ec2 delete-vpc-endpoints --vpc-endpoint-ids $eps 2>$null | Out-Null; _ok "Deleted VPC Endpoints" }
} catch { _skip "VPC Endpoint cleanup" $_ }

# 5) Internet Gateways
try {
  $igws = aws ec2 describe-internet-gateways --filters Name=attachment.vpc-id,Values=$VpcId `
    --query "InternetGateways[].InternetGatewayId" --output text 2>$null
  foreach($ig in ($igws -split "\s+" | ? {$_})) {
    try { aws ec2 detach-internet-gateway --internet-gateway-id $ig --vpc-id $VpcId 2>$null | Out-Null } catch {}
    try { aws ec2 delete-internet-gateway --internet-gateway-id $ig 2>$null | Out-Null } catch {}
  }
  if ($igws) { _ok "Deleted IGWs" }
} catch { _skip "IGW cleanup" $_ }

# 6) ENIs (detach if in-use, then delete)
try {
  $enis = aws ec2 describe-network-interfaces --filters Name=vpc-id,Values=$VpcId `
    --query "NetworkInterfaces[].{Id:NetworkInterfaceId,Att:Attachment.AttachmentId,Status:Status}" `
    --output json 2>$null | ConvertFrom-Json
  if ($enis) {
    foreach($eni in $enis){ if ($eni.Status -eq "in-use" -and $eni.Att) { try { aws ec2 detach-network-interface --attachment-id $eni.Att --force 2>$null | Out-Null } catch {} } }
    Start-Sleep 4
    foreach($eni in $enis){ try { aws ec2 delete-network-interface --network-interface-id $eni.Id 2>$null | Out-Null } catch {} }
    _ok "Deleted ENIs"
  }
} catch { _skip "ENI cleanup" $_ }

# 7) Subnets
try {
  $subs = aws ec2 describe-subnets --filters Name=vpc-id,Values=$VpcId --query "Subnets[].SubnetId" --output text 2>$null
  foreach($s in ($subs -split "\s+" | ? {$_})) { try { aws ec2 delete-subnet --subnet-id $s 2>$null | Out-Null } catch {} }
  if ($subs) { _ok "Deleted Subnets" }
} catch { _skip "Subnet cleanup" $_ }

# 8) Route Tables (non-main) — remove routes first, then delete with retries
try {
  $rtIds = aws ec2 describe-route-tables --filters Name=vpc-id,Values=$VpcId `
    --query "RouteTables[?Associations[?Main!=`true`]].RouteTableId" --output text 2>$null
  foreach($rt in ($rtIds -split "\s+" | ? {$_})) { Remove-NonLocalRoutesAndDelete-RT $rt }
} catch { _skip "Route Table cleanup" $_ }

# 9) NACLs (non-default)
try {
  $nacls = aws ec2 describe-network-acls --filters Name=vpc-id,Values=$VpcId `
    --query "NetworkAcls[?IsDefault==`false`].NetworkAclId" --output text 2>$null
  foreach($na in ($nacls -split "\s+" | ? {$_})) { try { aws ec2 delete-network-acl --network-acl-id $na 2>$null | Out-Null } catch {} }
  if ($nacls) { _ok "Deleted NACLs" }
} catch { _skip "NACL cleanup" $_ }

# 10) Security Groups (non-default)
try {
  $sgs = aws ec2 describe-security-groups --filters Name=vpc-id,Values=$VpcId `
    --query "SecurityGroups[?GroupName!='default'].GroupId" --output text 2>$null
  foreach($sg in ($sgs -split "\s+" | ? {$_})) { try { aws ec2 delete-security-group --group-id $sg 2>$null | Out-Null } catch {} }
  if ($sgs) { _ok "Deleted Security Groups" }
} catch { _skip "Security Group cleanup" $_ }

# 11) DHCP options (re-associate to default, then delete custom if orphaned)
try {
  $dhcpId = aws ec2 describe-vpcs --vpc-ids $VpcId --query "Vpcs[0].DhcpOptionsId" --output text 2>$null
  if ($dhcpId -and $dhcpId -ne "default") {
    try { aws ec2 associate-dhcp-options --vpc-id $VpcId --dhcp-options-id default 2>$null | Out-Null } catch {}
    # If no VPC still references this set, delete it.
    $still = aws ec2 describe-vpcs --filters Name=dhcp-options-id,Values=$dhcpId --query "Vpcs[].VpcId" --output text 2>$null
    if (-not $still) { try { aws ec2 delete-dhcp-options --dhcp-options-id $dhcpId 2>$null | Out-Null } catch {} }
  }
} catch { _skip "DHCP options cleanup" $_ }

# 12) Delete the VPC
try { aws ec2 delete-vpc --vpc-id $VpcId 2>$null | Out-Null; _ok "Deleted VPC $VpcId" }
catch { _skip "Delete VPC $VpcId" $_ }

Write-Host "==> VPC purge done." -ForegroundColor Cyan
