# IWriting browser launcher
# Waits for port 7860 to be open, then opens it in default browser
# Auto-exits once browser is opened

$port = 7860
$url = "http://127.0.0.1:$port"
$maxAttempts = 90
$attempt = 0

Write-Host "Waiting for Gradio on $url ..."

while ($attempt -lt $maxAttempts) {
    $attempt++
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", $port)
        $client.Close()
        Write-Host "Gradio is up! Opening browser ..."
        Start-Process $url
        exit 0
    } catch {
        Start-Sleep -Seconds 1
    }
}

Write-Host "[TIMEOUT] $url did not respond within $maxAttempts seconds."
Write-Host "Please open $url manually after Gradio finishes loading."
Read-Host "Press Enter to close"
exit 1
