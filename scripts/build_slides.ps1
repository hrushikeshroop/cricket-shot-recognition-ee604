param([string]$SpecPath = "slides/content.json")
# Create an editable presentation with native PowerPoint text, tables and diagrams.
# Open and export without a presentation window. Run outside the sandbox because
# Office COM cannot attach to a user logon session inside the sandbox.
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$spec = Get-Content -LiteralPath (Join-Path $workspace $SpecPath) -Raw -Encoding UTF8 | ConvertFrom-Json
$outDir = Join-Path $workspace 'output/presentation'
$buildDir = Join-Path $workspace '.build/presentation'
$previewDir = Join-Path $buildDir 'previews'
New-Item -ItemType Directory -Force -Path $outDir,$buildDir,$previewDir | Out-Null

function RGB([string]$hex) {
    $v = $hex.TrimStart('#')
    return [Convert]::ToInt32($v.Substring(0,2),16) + 256*[Convert]::ToInt32($v.Substring(2,2),16) + 65536*[Convert]::ToInt32($v.Substring(4,2),16)
}
$ink = RGB '20382A'; $green = RGB '426B42'; $muted = RGB '72806B'; $paper = RGB 'F7F8F1'; $orange = RGB 'AF623E'

function Text($slide,[string]$text,[double]$x,[double]$y,[double]$w,[double]$h,[double]$size=18,[bool]$bold=$false,[int]$color=$ink) {
    $shape=$slide.Shapes.AddTextbox(1,[single]$x,[single]$y,[single]$w,[single]$h)
    $shape.TextFrame.MarginLeft=0; $shape.TextFrame.MarginRight=0
    $shape.TextFrame.MarginTop=0; $shape.TextFrame.MarginBottom=0
    $shape.TextFrame.WordWrap=-1; $shape.TextFrame.AutoSize=0
    $range=$shape.TextFrame.TextRange; $range.Text=$text
    $range.Font.Name='Arial'; $range.Font.Size=$size; $range.Font.Color.RGB=$color
    $range.Font.Bold=$(if($bold){-1}else{0})
    $range.ParagraphFormat.SpaceAfter=8
    $shape.TextFrame2.AutoSize=0
    $shape.TextFrame2.WordWrap=-1
    $shape.Height=[single]$h
    return $shape
}

function Picture($slide,[string]$relative,[double]$x,[double]$y,[double]$w,[double]$h) {
    $path=Join-Path $workspace $relative
    if (!(Test-Path -LiteralPath $path)) {throw "Missing slide asset: $relative"}
    Add-Type -AssemblyName System.Drawing
    $image=[System.Drawing.Image]::FromFile($path)
    try {$ratio=[double]$image.Width/$image.Height} finally {$image.Dispose()}
    $width=[Math]::Min($w,$h*$ratio);$height=$width/$ratio
    $slide.Shapes.AddPicture($path,0,-1,[single]($x+($w-$width)/2),[single]($y+($h-$height)/2),[single]$width,[single]$height) | Out-Null
}

function NativeTable($slide,$tableSpec) {
    $rows=@($tableSpec.rows); $headers=@($tableSpec.headers)
    $shape=$slide.Shapes.AddTable($rows.Count+1,$headers.Count,[single]$tableSpec.x,[single]$tableSpec.y,[single]$tableSpec.w,[single]$tableSpec.h)
    $table=$shape.Table
    if($tableSpec.widths){for($j=0;$j -lt $headers.Count;$j++){$table.Columns.Item($j+1).Width=[single]$tableSpec.widths[$j]}}
    for($i=0;$i -le $rows.Count;$i++) {
        for($j=0;$j -lt $headers.Count;$j++) {
            $cell=$table.Cell($i+1,$j+1)
            $val=if($i -eq 0){[string]$headers[$j]}else{[string]$rows[$i-1][$j]}
            $cell.Shape.TextFrame.TextRange.Text=$val
            $cell.Shape.TextFrame.MarginLeft=9; $cell.Shape.TextFrame.MarginRight=6
            $cell.Shape.TextFrame.MarginTop=5; $cell.Shape.TextFrame.MarginBottom=4
            $cell.Shape.TextFrame.VerticalAnchor=3
            $font=$cell.Shape.TextFrame.TextRange.Font
            $font.Name='Arial';$font.Size=[double]$tableSpec.fontSize
            if($i -eq 0){$cell.Shape.Fill.ForeColor.RGB=$ink;$font.Color.RGB=RGB 'FFFFFF';$font.Bold=-1}
            else {
                $cell.Shape.Fill.ForeColor.RGB=$(if($i%2 -eq 0){RGB 'EEF2E7'}else{RGB 'FFFFFF'})
                $font.Color.RGB=$ink;$font.Bold=0
                if($tableSpec.heatmap -and $j -gt 0){
                    $den=[double]$tableSpec.totals[$i-1];$fraction=if($den -gt 0){[double]$val/$den}else{0}
                    $rr=[int](246-170*$fraction);$gg=[int](248-135*$fraction);$bb=[int](238-165*$fraction)
                    $cell.Shape.Fill.ForeColor.RGB=$rr+256*$gg+65536*$bb
                    if($fraction -gt .6){$font.Color.RGB=RGB 'FFFFFF'}
                }
            }
            for($b=1;$b -le 4;$b++){$cell.Borders.Item($b).ForeColor.RGB=RGB 'D8DFD0';$cell.Borders.Item($b).Weight=.5}
        }
    }
}

$app=$null;$deck=$null
try {
    $app=New-Object -ComObject PowerPoint.Application
    $deck=$app.Presentations.Add(0)
    $deck.PageSetup.SlideWidth=960;$deck.PageSetup.SlideHeight=540
    foreach($entry in $spec.slides) {
        $slide=$deck.Slides.Add($deck.Slides.Count+1,12)
        $slide.FollowMasterBackground=0;$slide.Background.Fill.Solid();$slide.Background.Fill.ForeColor.RGB=$paper
        if($entry.kind -eq 'cover') {
            Text $slide $entry.kicker 48 45 800 24 11 $true $green | Out-Null
            Text $slide $entry.title 48 115 500 185 44 $true $ink | Out-Null
            Text $slide $entry.subtitle 50 323 440 85 20 $false $muted | Out-Null
            Picture $slide $entry.image 565 100 350 315
            if($entry.team_text) {
                $teamShape=Text $slide $entry.team_text 50 423 850 68 12 $false $ink
                $teamShape.TextFrame.TextRange.ParagraphFormat.SpaceAfter=2
                Text $slide $entry.bottom 50 506 850 19 10 $false $muted | Out-Null
            } else {
                Text $slide $entry.bottom 50 473 850 25 11 $false $muted | Out-Null
            }
        } else {
            Text $slide $entry.title 48 32 865 48 32 $true $ink | Out-Null
            if($entry.subtitle){Text $slide $entry.subtitle 49 90 855 44 15 $false $muted | Out-Null}
            if($entry.text){Text $slide ($entry.text -join "`r`r") 50 145 850 310 20 $false $ink | Out-Null}
            foreach($block in $entry.blocks){
                Text $slide $block.text $block.x $block.y $block.w $block.h $block.size ([bool]$block.bold) $(if($block.color){RGB $block.color}else{$ink}) | Out-Null
            }
            foreach($imageSpec in $entry.images){Picture $slide $imageSpec.path $imageSpec.x $imageSpec.y $imageSpec.w $imageSpec.h}
            foreach($tableSpec in $entry.tables){NativeTable $slide $tableSpec}
            if($entry.diagram){
                $boxes=@($entry.diagram)
                for($i=0;$i -lt $boxes.Count;$i++){
                    $xx=50+$i*176
                    $box=$slide.Shapes.AddShape(1,[single]$xx,176,155,115)
                    $box.Fill.ForeColor.RGB=RGB 'E7EEDD';$box.Line.ForeColor.RGB=RGB 'B3C3A6';$box.Line.Weight=1
                    Text $slide $boxes[$i] ($xx+12) 194 132 85 17 $true $ink | Out-Null
                    if($i -lt $boxes.Count-1){$line=$slide.Shapes.AddLine([single]($xx+158),233,[single]($xx+173),233);$line.Line.EndArrowheadStyle=3;$line.Line.ForeColor.RGB=$green}
                }
            }
            if($entry.callout){Text $slide $entry.callout 50 449 855 42 16 $false $orange | Out-Null}
            Text $slide 'CRICKET SHOT RECOGNITION / EE604' 50 511 750 15 8 $false $muted | Out-Null
            Text $slide ([string]$slide.SlideIndex) 883 508 25 18 10 $false $muted | Out-Null
        }
        if($entry.notes){$slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=[string]$entry.notes}
    }
    $pptx=Join-Path $outDir 'CricketShot_Presentation.pptx'
    $pdf=Join-Path $buildDir 'CricketShot_Presentation.pdf'
    $deck.SaveAs($pptx,24)
    $deck.SaveAs($pdf,32)
    $deck.Export($previewDir,'PNG',1600,900)
    # Verify text geometry after the actual Office renderer has laid out all text.
    $issues=@()
    foreach($slide in $deck.Slides){
        foreach($shape in $slide.Shapes){
            if($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1){
                $range=$shape.TextFrame.TextRange
                if($range.BoundHeight -gt ($shape.Height+2)){$issues+=@{slide=$slide.SlideIndex;shape=$shape.Name;issue='Text height exceeds shape';height=$shape.Height;boundHeight=$range.BoundHeight;text=$range.Text}}
                if($shape.Left -lt -1 -or $shape.Top -lt -1 -or $shape.Left+$shape.Width -gt 962 -or $shape.Top+$shape.Height -gt 542){$issues+=@{slide=$slide.SlideIndex;shape=$shape.Name;issue='Shape outside slide'}}
            }
        }
    }
    $record=@{renderer='Microsoft PowerPoint 16 COM';slide_count=$deck.Slides.Count;issues=@($issues);font='Arial';source='slides/content.json'}
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $workspace 'results/presentation_validation.json') -Encoding UTF8
    Write-Output "Saved $($deck.Slides.Count) editable slides. Geometry issues: $($issues.Count)"
} finally {
    if($deck){$deck.Close();[System.Runtime.InteropServices.Marshal]::ReleaseComObject($deck)|Out-Null}
    if($app){$app.Quit();[System.Runtime.InteropServices.Marshal]::ReleaseComObject($app)|Out-Null}
    [GC]::Collect();[GC]::WaitForPendingFinalizers()
}
