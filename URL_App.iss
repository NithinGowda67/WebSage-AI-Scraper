[Setup]
AppName=URL Analysis and Comparison
AppVersion=1.0
DefaultDirName={pf}\URL
DefaultGroupName=URL
UninstallDisplayIcon={app}\app.ico
OutputDir=..\
OutputBaseFilename=URL
Compression=lzma
SolidCompression=yes

[Files]
Source: "app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "install_and_run.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "pc_req.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "users.json"; DestDir: "{app}"; Flags: ignoreversion
Source: ".env"; DestDir: "{app}"; Flags: ignoreversion

Source: "optuna_models\*"; DestDir: "{app}\optuna_models"; Flags: recursesubdirs createallsubdirs
Source: "runs\*"; DestDir: "{app}\runs"; Flags: recursesubdirs createallsubdirs
Source: ".streamlit\*"; DestDir: "{app}\.streamlit"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Run URL Analysis and Comparison"; Filename: "{app}\install_and_run.bat"

[UninstallDelete]
Type: files; Name: "{app}\venv"