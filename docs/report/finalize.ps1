# Open the built report in Word, refresh the table of contents and the figure/table lists,
# save, and export a PDF copy next to it. Usage: pwsh docs/report/finalize.ps1
$docx = Join-Path (Split-Path $PSScriptRoot -Parent) "Bao_cao_KG4RS_RippleNet_CKAN.docx"
$pdf = [System.IO.Path]::ChangeExtension($docx, ".pdf")

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $false)
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    $pages = $doc.ComputeStatistics(2)
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)
    $doc.Close($false)
    "pages: $pages"
    "saved: $docx"
    "pdf:   $pdf"
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
