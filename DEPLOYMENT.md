# 학생부 탐색기 Windows 배포

## 교사에게 보이는 실행 방식

설치 후 **학생부 탐색기** 바로가기를 실행하면 PC 내부의 `127.0.0.1` 주소에서 앱이 시작되고 기본 웹브라우저가 열립니다. 외부 웹 서버에 접속하는 방식이 아니며, 현재 Streamlit 화면과 CSS가 그대로 표시됩니다.

앱을 완전히 끄려면 왼쪽 사이드바 아래의 **프로그램 종료**를 누릅니다. 브라우저 탭만 닫으면 로컬 앱은 백그라운드에서 계속 실행될 수 있습니다.

사용자 사전, 내장 학과 DB와 로그는 다음 위치에 저장됩니다.

```text
%LOCALAPPDATA%\StudentRecordExplorer
```

## 제작자용 빌드

Windows에서 `build_exe.bat`을 실행합니다. Python 가상환경과 패키징 도구는 제작 PC에만 필요하며, 교사 PC에는 Python이 필요하지 않습니다.

생성 결과:

- `dist\StudentRecordExplorer-1.0.2\StudentRecordExplorer.exe`: 빌드 확인용 실행 파일
- `release\StudentRecordExplorer-Portable-1.0.2.zip`: 무설치 압축본
- `release\StudentRecordExplorer-Setup-1.0.2.exe`: Inno Setup이 설치된 경우 생성되는 설치본

제품 버전은 저장소 루트의 `VERSION` 파일, Windows 실행 파일 정보, 설치 프로그램과 배포 파일명에서 `1.0.2`로 통일합니다.

정식 배포 전에는 Python이 없는 별도 Windows PC, 관리자 권한이 없는 계정, Windows Defender가 활성화된 환경에서 확인합니다. 공개 배포 시에는 코드 서명을 권장합니다.
