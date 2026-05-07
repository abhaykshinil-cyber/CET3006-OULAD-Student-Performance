$ErrorActionPreference = "Stop"

function Normalize-Text {
    param([string]$Text)
    if ($null -eq $Text) { return "" }
    return $Text
}

function Insert-Paragraph {
    param(
        $Selection,
        [string]$Text,
        [string]$Style = "Normal",
        [int]$Alignment = 0,
        [bool]$Italic = $false,
        [int]$FontSize = 11,
        [bool]$Bold = $false
    )
    $Selection.Style = $Style
    $Selection.ParagraphFormat.Alignment = $Alignment
    $Selection.Font.Italic = [int]$Italic
    $Selection.Font.Bold = [int]$Bold
    $Selection.Font.Size = $FontSize
    $Selection.TypeText((Normalize-Text $Text))
    $Selection.TypeParagraph()
    $Selection.Font.Italic = 0
    $Selection.Font.Bold = 0
}

function Insert-Bullet {
    param($Selection, [string]$Text)
    $Selection.Style = "Normal"
    $Selection.ParagraphFormat.Alignment = 0
    $Selection.TypeText((Normalize-Text $Text))
    $Selection.TypeParagraph()
    $Selection.MoveUp()
    $Selection.Range.ListFormat.ApplyBulletDefault() | Out-Null
    $Selection.MoveDown()
}

function Insert-Numbered {
    param($Selection, [string]$Text)
    $Selection.Style = "Normal"
    $Selection.ParagraphFormat.Alignment = 0
    $Selection.TypeText((Normalize-Text $Text))
    $Selection.TypeParagraph()
    $Selection.MoveUp()
    $Selection.Range.ListFormat.ApplyNumberDefault() | Out-Null
    $Selection.MoveDown()
}

function Insert-Image {
    param($Selection, [string]$Path)
    $Selection.ParagraphFormat.Alignment = 1
    $Selection.InlineShapes.AddPicture($Path) | Out-Null
    $Selection.TypeParagraph()
    $Selection.ParagraphFormat.Alignment = 0
}

function Parse-TableLine {
    param([string]$Line)
    $trimmed = $Line.Trim()
    $trimmed = $trimmed.Trim('|')
    return ($trimmed -split '\|') | ForEach-Object { (Normalize-Text $_.Trim()) }
}

function Insert-Table {
    param($Document, $Selection, [System.Collections.Generic.List[string]]$Lines)
    if ($Lines.Count -lt 2) { return }

    $rows = @()
    foreach ($line in $Lines) {
        if ($line -match '^\|\s*-') { continue }
        $rows += ,(Parse-TableLine $line)
    }
    if ($rows.Count -eq 0) { return }

    $rowCount = $rows.Count
    $colCount = $rows[0].Count
    $table = $Document.Tables.Add($Selection.Range, $rowCount, $colCount)
    $table.Borders.Enable = 1
    $table.Range.Font.Size = 10
    $table.Rows.Alignment = 1
    for ($r = 1; $r -le $rowCount; $r++) {
        for ($c = 1; $c -le $colCount; $c++) {
            $text = ""
            if ($c -le $rows[$r - 1].Count) { $text = $rows[$r - 1][$c - 1] }
            $table.Cell($r, $c).Range.Text = $text
        }
    }
    $table.Rows.Item(1).Range.Bold = 1
    $Selection.MoveDown() | Out-Null
    $Selection.TypeParagraph()
}

$root = "C:\Users\abhay\OneDrive\Documents\RESEARCH PAPER"
$outputDoc = Join-Path $root "docs\CET3006_final_submission.docx"
$markdownPath = Join-Path $root "docs\CET3006_research_paper_aligned.md"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    $doc = $word.Documents.Add()
    $sel = $word.Selection

    Insert-Paragraph -Selection $sel -Text "CET3006 Research Project" -Style "Title" -Alignment 1 -Bold $true -FontSize 18
    Insert-Paragraph -Selection $sel -Text "ABHAY KALATHIL SHINIL" -Style "Normal" -Alignment 1 -FontSize 14
    Insert-Paragraph -Selection $sel -Text "<Registration Number>" -Style "Normal" -Alignment 1 -FontSize 12
    Insert-Paragraph -Selection $sel -Text "<Degree Programme>" -Style "Normal" -Alignment 1 -FontSize 12
    Insert-Paragraph -Selection $sel -Text "" -Style "Normal" -Alignment 1
    Insert-Paragraph -Selection $sel -Text "Accuracy, Interpretability, and Efficiency in Student Performance Prediction on OULAD: A Comparative Study of Random Forest, XGBoost, TabNet, and FT-Transformer" -Style "Heading 1" -Alignment 1
    Insert-Paragraph -Selection $sel -Text "Word count 5000" -Style "Normal" -Alignment 1 -FontSize 12
    $sel.InsertBreak(7) | Out-Null

    $lines = [System.Collections.Generic.List[string]](Get-Content $markdownPath)
    $i = 0
    while ($i -lt $lines.Count) {
        $line = $lines[$i]
        $trim = $line.Trim()

        if ($trim -eq "") {
            $sel.TypeParagraph()
            $i++
            continue
        }

        if ($trim.StartsWith("# ")) {
            $i++
            continue
        }

        if ($trim.StartsWith("## ")) {
            Insert-Paragraph -Selection $sel -Text $trim.Substring(3) -Style "Heading 1" -FontSize 14 -Bold $true
            $i++
            continue
        }

        if ($trim.StartsWith("### ")) {
            Insert-Paragraph -Selection $sel -Text $trim.Substring(4) -Style "Heading 2" -FontSize 12 -Bold $true
            $i++
            continue
        }

        if ($trim -match '^!\[.*\]\((.+)\)$') {
            $img = $Matches[1]
            $imgPath = [System.IO.Path]::GetFullPath((Join-Path (Split-Path $markdownPath -Parent) $img))
            if (Test-Path $imgPath) {
                Insert-Image -Selection $sel -Path $imgPath
            }
            $i++
            continue
        }

        if ($trim -match '^\*Figure .*?\*$') {
            $caption = $trim.Trim('*')
            Insert-Paragraph -Selection $sel -Text $caption -Style "Normal" -Alignment 1 -Italic $true -FontSize 10
            $i++
            continue
        }

        if ($trim -like "- *") {
            Insert-Bullet -Selection $sel -Text $trim.Substring(2)
            $i++
            continue
        }

        if ($trim -match '^\d+\.\s+') {
            $text = $trim -replace '^\d+\.\s+', ''
            Insert-Numbered -Selection $sel -Text $text
            $i++
            continue
        }

        if ($trim.StartsWith("|")) {
            $tableLines = [System.Collections.Generic.List[string]]::new()
            while ($i -lt $lines.Count -and $lines[$i].Trim().StartsWith("|")) {
                $tableLines.Add($lines[$i])
                $i++
            }
            Insert-Table -Document $doc -Selection $sel -Lines $tableLines
            continue
        }

        Insert-Paragraph -Selection $sel -Text $trim -Style "Normal" -FontSize 11
        $i++
    }

    $doc.SaveAs2([string]$outputDoc, 16)
    $doc.Close()
    Write-Output $outputDoc
}
finally {
    $word.Quit()
}
