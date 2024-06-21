Function Find-NewName {
    Param(
        [Parameter(Mandatory)]
        [string]$Path
    )
    $i = 0
    While (Test-Path $Path) {
        $i++
        $PathItem = Get-Item $Path
        $Path = "$($PathItem.Directoryname)\$($PathItem.Basename).$i.md"
    }
    $Path
}

rm -r .\all\*
git submodule sync
Get-ChildItem .\TCL\docs\* -Include *.md -Exclude _* -Recurse | % { Copy-Item -Path $_ -Destination (Find-NewName ".\all\$((Get-Item $_).Name)") }
python clean.py all
