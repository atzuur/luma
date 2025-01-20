Function Find-NewName {
    Param(
        [Parameter(Mandatory)]
        [string]$Path
    )
    $i = 0
    while (Test-Path $Path) {
        $i++
        $PathItem = Get-Item $Path
        $Path = "$($PathItem.Directoryname)\$($PathItem.Basename).$i.md"
    }
    $Path
}

if (-not (Test-Path .\TCL)) {
    git clone https://github.com/KQM-git/TCL.git
} 

if (Test-Path .\all) {
    rm -r .\all\*
} else {
    mkdir .\all
}

Get-ChildItem .\TCL\docs\* -Include *.md -Exclude _* -Recurse | % {
    Copy-Item -Path $_ -Destination (Find-NewName ".\all\$((Get-Item $_).Name)")
}
python clean_tcl_md.py all
rmdir -r .\all
