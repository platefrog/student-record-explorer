#define MyAppName "학생부 탐색기"
#define MyAppEnglishName "StudentRecord Explorer"
#define MyAppVersion "1.0.2"
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
InfoBeforeFile=installer_info_ko.txt
LicenseFile=installer_agreement_ko.txt
UsePreviousAppDir=yes

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 바로가기:"; Flags: unchecked

[Files]
Source: "dist\StudentRecordExplorer-{#MyAppVersion}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; 중단되거나 수동 삭제된 구버전의 잔여 파일도 새 설치 전에 정리한다.
Type: filesandordirs; Name: "{app}\_internal"
Type: files; Name: "{app}\{#MyAppExeName}"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{#MyAppName} 실행"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; 실행 중인 앱 때문에 삭제가 실패하지 않도록 먼저 종료한다.
Filename: "{sys}\taskkill.exe"; Parameters: "/F /T /IM {#MyAppExeName}"; Flags: runhidden waituntilterminated; RunOnceId: "StopStudentRecordExplorer"

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  { 구버전이 실행 중이어도 사용자가 작업 관리자에서 직접 종료할 필요가 없게 한다. }
  Exec(ExpandConstant('{sys}\taskkill.exe'),
    '/F /T /IM {#MyAppExeName}', '', SW_HIDE,
    ewWaitUntilTerminated, ResultCode);
  Sleep(500);
  Result := '';
end;
