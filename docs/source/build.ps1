Try {
    sphinx-build -M html ./ ./ -Evc ./
} catch [System.Management.Automation.CommandNotFoundException] {
    Write-Host "'sphinx-build' command not found, probably in wrong environment."
    exit
}

Get-ChildItem ../ -Exclude source | Remove-Item -Recurse

Get-ChildItem ./html/ | Move-Item -Destination ../

Remove-Item ./html