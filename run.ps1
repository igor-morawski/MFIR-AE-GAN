Invoke-Expression -Command "conda activate tf2-gpu"

$list_fp = "tmp/list.txt"
if (!(Test-Path $list_fp)) {New-Item -itemType File -Path $list_fp}
$list = Get-Content $list_fp

$epochs = 500

$norms = ("interframe_minmax", "est_minmax", "zscore")
#$norms = ("zscore")
$lrs = ("1e-5")
$params = @{
    "norm" = @($norms)
    "lr" = @($lrs)
    "mins" = (20, 25, 27)
    "maxs" = (40, 35, 33)
 }
$mins = $params['mins']
$maxs = $params['maxs']
foreach ($norm in $params['norm']) {
    foreach ($lr in $params['lr']) {
        $minmax_cmds = New-Object System.Collections.Generic.List[System.Object]
        if ($norm -eq "est_minmax") {foreach ($_ in 0..($mins.length-1)) {$min = $mins[$_]; $max = $maxs[$_]; $minmax_cmds.Add("--min=$min --max=$max")}} else {$minmax_cmds += ""}
        foreach ($minmax_cmd in $minmax_cmds) {
            $minmax_info=$minmax_cmd
            $minmax_info=$minmax_info.replace('--','')
            $minmax_info=$minmax_info.replace('=','')
            $minmax_info=$minmax_info.replace('min','')
            $minmax_info=$minmax_info.replace('max','')
            $minmax_info=$minmax_info.replace(' ','_')
            $prefix = "$norm"+"$minmax_info"+"_"+"$lr"+"_"
            $suffix = ""
            $cmd = "python convVAE.py --norm=$norm --lr=$lr --epochs=$epochs $minmax_cmd --prefix=$prefix "; $cmd;
            If (-Not ($list | Select-String $cmd)) {$cmd; Invoke-Expression -Command $cmd; Add-Content $list_fp $cmd;}
        }
    }
}