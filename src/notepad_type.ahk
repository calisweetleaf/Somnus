#Requires AutoHotkey v2.0
; Usage: AutoHotkey.exe notepad_type.ahk "Your text here" "C:\path\to\file.txt"
text := A_Args[1]           ; The first argument: text to type.
filePath := ""
if A_Args.Length() > 1 {
    filePath := A_Args[2]   ; The second argument: file path to save.
}

Run, notepad.exe
WinWaitActive, ahk_exe notepad.exe, , 3        ; Wait up to 3 seconds for Notepad to be active
if ErrorLevel {
    MsgBox, 16, Error, Failed to open Notepad.
    ExitApp
}
; Type the text into Notepad
SendRaw % text

if (filePath != "") {
    ; Save the file if path provided
    Send, ^s     ; Ctrl+S triggers Save dialog
    WinWaitActive, Save As, , 2
    if ErrorLevel {
        MsgBox, 16, Error, Save dialog did not appear.
        ExitApp
    }
    ; Type the file path and press Enter
    SendRaw % filePath
    Send, {Enter}
    ; Wait a moment for file to save
    Sleep, 500
}
ExitApp