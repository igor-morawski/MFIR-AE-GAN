Invoke-Expression -Command "conda activate tf2-gpu"

$list_fp = "tmp/list.txt"
if (!(Test-Path $list_fp)) {New-Item -itemType File -Path $list_fp}
$list = Get-Content $list_fp

$epochs = 200

$norms = ("interframe_minmax", "zscore")
$lrs = ("1e-4", "1e-5", "1e-6")
$latent_dims = ("100", "250", "500", "750", "1000")
$params = @{
    "norm" = @($norms)
    "lr" = @($lrs)
    "latent_dims" = @($latent_dims)
    #"mins" = (20, 25, 27)
    #"maxs" = (40, 35, 33)
 }
$mins = $params['mins']
$maxs = $params['maxs']

foreach ($latent_dim in $params['latent_dims']) {
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
                    $prefix = "ld"+"$latent_dim"+"_"+"$norm"+"$minmax_info"+"_"+"$lr"+"_"
                    $suffix = ""
                    $cmd = "python convVAE.py --norm=$norm --lr=$lr --latent_dim=$latent_dim --epochs=$epochs $minmax_cmd --prefix=$prefix "; $cmd;
                    If (-Not ($list | Select-String $cmd)) {$cmd; Invoke-Expression -Command $cmd; Add-Content $list_fp $cmd;}
                }
            }
    }
}

[console]::beep(500,300);[console]::beep(1000,300);[console]::beep(500,300);[console]::beep(1000,300);
[console]::beep(500,300);[console]::beep(1000,300);[console]::beep(500,300);[console]::beep(1000,300);
[console]::beep(500,300);[console]::beep(1000,300);[console]::beep(500,300);[console]::beep(1000,300);