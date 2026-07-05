#define MyAppName "학생부 탐색기"
#define MyAppEnglishName "StudentRecord Explorer"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Park Hyojin"
#define MyAppExeName "StudentRecordExplorer.exe"

[Setup]
AppId={{4CF71DBE-2D96-4E91-81F1-C7C76A7AA8A3}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\StudentRecordExplorer
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=release
OutputBaseFilename=StudentRecordExplorer-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#MyAppName}
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 바로가기:"; Flags: unchecked

[Files]
Source: "dist\StudentRecordExplorer-{#MyAppVersion}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{#MyAppName} 실행"; Flags: nowait postinstall skipifsilent
