param([int]$Port = 6040)
# Render the local project in isolated headless profiles for visual QA.
# No user browser profile, account or extension is accessed.
$ErrorActionPreference='Stop'
$workspace=Split-Path -Parent $PSScriptRoot
$chrome='C:/Program Files/Google/Chrome/Application/chrome.exe'
if(!(Test-Path -LiteralPath $chrome)){throw 'Install Chrome or edit this renderer path. The app itself does not require Chrome.'}
$examples=Get-Content -LiteralPath (Join-Path $workspace 'demo/image_examples.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$first=$examples[0].id
$failure=($examples | Where-Object {$_.note -like 'Confident failure*'} | Select-Object -First 1).id
$runToken=[guid]::NewGuid().ToString('N').Substring(0,8)
$cases=@(
    @{name='ui_home';size='1440,1250';url="http://127.0.0.1:$Port/"},
    @{name='ui_image_prediction';size='1440,680';url="http://127.0.0.1:$Port/?example=$first&focus=1"},
    @{name='ui_failure';size='1440,680';url="http://127.0.0.1:$Port/?example=$failure&focus=1"},
    @{name='ui_narrow';size='600,1200';url="http://127.0.0.1:$Port/?example=$first&focus=1"}
)
foreach($case in $cases){
    $profile=Join-Path $workspace ('.build/headless-revised-'+$runToken+'-'+$case.name)
    $target=Join-Path $workspace ('results/'+$case.name+'.png')
    $log=Join-Path $workspace ('.build/'+$case.name+'.stderr.log')
    $arguments=@('--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--hide-scrollbars',
                 ('--window-size='+$case.size),'--virtual-time-budget=12000',
                 ('--user-data-dir="'+$profile+'"'),('--screenshot="'+$target+'"'),('"'+$case.url+'"'))
    $started=Get-Date
    Start-Process -FilePath $chrome -ArgumentList $arguments -WindowStyle Hidden -RedirectStandardError $log
    # Office/Chrome process handles can be inaccessible from this host. Require
    # an actual newly written screenshot instead of treating process exit as proof.
    $fresh=$false
    $deadline=(Get-Date).AddSeconds(45)
    while((Get-Date) -lt $deadline){
        if((Test-Path -LiteralPath $target) -and ((Get-Item -LiteralPath $target).LastWriteTime -ge $started)){$fresh=$true;break}
        Start-Sleep -Milliseconds 250
    }
    if(!$fresh){throw ('No fresh screenshot was generated: '+$case.name)}
    if(!(Test-Path -LiteralPath $target)){throw ('Screenshot missing: '+$case.name)}
    Write-Output ('Rendered '+$case.name)
}
