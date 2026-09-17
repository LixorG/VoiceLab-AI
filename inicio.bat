@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: ============================================================
::  VoiceLab AI - Inicio rapido
::  Doble clic para iniciar. Elige modo produccion o desarrollo.
:: ============================================================

title VoiceLab AI - Iniciando...

:: Ubicacion raiz del proyecto (carpeta donde esta este .bat)
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "PY=%ROOT%\backend\.venv\Scripts\python.exe"
set "NPM_CMD=npm"
set "FRONTEND=%ROOT%\frontend"
set "DIST=%ROOT%\frontend\dist\index.html"
set "BACKEND_URL=http://127.0.0.1:8000"
set "DEV_URL=http://127.0.0.1:5173"

:: -- Verificar instalacion -------------------------------------
if not exist "%PY%" (
    echo.
    echo  [ERROR] El entorno virtual no existe.
    echo          Ejecuta primero install.cmd para instalar VoiceLab AI.
    echo.
    pause
    exit /b 1
)

:: -- Menu de inicio --------------------------------------------
:MENU
cls
echo.
echo  ====================================================
echo             VoiceLab AI - Inicio Rapido
echo  ====================================================
echo.
echo   [1]  Modo Produccion
echo        Backend + Frontend compilado
echo        URL: http://127.0.0.1:8000
echo.
echo   [2]  Modo Desarrollo (hot-reload)
echo        Backend en :8000 + Vite en :5173
echo        URL: http://127.0.0.1:5173
echo.
echo   [3]  Salir
echo.
echo  ====================================================
echo.
set /p "OPCION=  Elige una opcion (1/2/3): "

if "%OPCION%"=="1" goto PRODUCCION
if "%OPCION%"=="2" goto DESARROLLO
if "%OPCION%"=="3" exit /b 0
echo  Opcion no valida. Intentalo de nuevo.
timeout /t 2 >nul
goto MENU

:: ------------------------------------------------------------
:PRODUCCION
title VoiceLab AI - Modo Produccion
cls
echo.
echo  [INFO] Iniciando en modo Produccion...
echo  [INFO] URL: %BACKEND_URL%
echo.

:: Verificar si el puerto 8000 ya esta en uso
netstat -ano | findstr ":8000 " | findstr "LISTEN" >nul 2>&1
if not errorlevel 1 (
    echo  [AVISO] El puerto 8000 ya esta en uso. Esta VoiceLab AI ya abierto?
    echo          Cierra la instancia anterior o abre http://127.0.0.1:8000
    echo.
    pause
    goto MENU
)

:: Compilar frontend si no existe o esta desactualizado
if not exist "%DIST%" (
    echo  [INFO] Compilando la interfaz por primera vez...
    call :COMPILAR_FRONTEND
    if errorlevel 1 (
        echo  [ERROR] No se pudo compilar la interfaz.
        pause
        goto MENU
    )
)

:: Abrir navegador cuando el servidor responda
start "" /B cmd /c "timeout /t 5 >nul && curl -s --retry 60 --retry-delay 1 --retry-connrefused %BACKEND_URL%/api/system/health >nul 2>&1 && start %BACKEND_URL%"

:: Iniciar backend (sirve tambien el frontend compilado)
echo  [OK] Servidor iniciado. Abriendo navegador en cuanto este listo...
echo       Presiona Ctrl+C para detener.
echo.
cd /d "%ROOT%"
"%PY%" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --log-level warning
goto FIN

:: ------------------------------------------------------------
:DESARROLLO
title VoiceLab AI - Modo Desarrollo
cls
echo.
echo  [INFO] Iniciando en modo Desarrollo...
echo  [INFO] Backend : http://127.0.0.1:8000
echo  [INFO] Frontend: http://127.0.0.1:5173  (hot-reload activo)
echo.

:: Verificar que npm este disponible
where npm >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] npm no encontrado. Instala Node.js desde https://nodejs.org
    pause
    goto MENU
)

:: Verificar node_modules
if not exist "%FRONTEND%\node_modules\" (
    echo  [INFO] Instalando dependencias del frontend...
    pushd "%FRONTEND%"
    npm install
    popd
)

:: Iniciar backend en ventana separada con hot-reload
start "VoiceLab - Backend :8000" cmd /k "title VoiceLab - Backend :8000 && cd /d "%ROOT%" && "%PY%" -m uvicorn app.main:app --app-dir backend --reload --reload-dir backend/app --host 127.0.0.1 --port 8000"

:: Esperar un momento antes de iniciar el frontend
timeout /t 2 >nul

:: Iniciar frontend Vite en ventana separada
start "VoiceLab - Frontend :5173" cmd /k "title VoiceLab - Frontend :5173 && cd /d "%FRONTEND%" && npm run dev"

:: Abrir navegador cuando Vite este listo
echo  [INFO] Esperando que los servidores esten listos...
timeout /t 4 >nul

:: Abrir el navegador en la URL de desarrollo
start "" "%DEV_URL%"

echo.
echo  [OK] VoiceLab AI iniciado en modo desarrollo.
echo       Backend : http://127.0.0.1:8000
echo       Frontend: http://127.0.0.1:5173
echo.
echo  Cierra las ventanas de Backend y Frontend para detener los servidores.
echo.
pause
goto FIN

:: ------------------------------------------------------------
:COMPILAR_FRONTEND
where npm >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] npm no encontrado. Instala Node.js para compilar el frontend.
    exit /b 1
)
if not exist "%FRONTEND%\node_modules\" (
    echo  [INFO] Instalando dependencias del frontend...
    pushd "%FRONTEND%"
    npm install
    if errorlevel 1 (popd & exit /b 1)
    popd
)
pushd "%FRONTEND%"
echo  [INFO] Compilando frontend (npm run build)...
npm run build
set "BUILD_ERR=%ERRORLEVEL%"
popd
exit /b %BUILD_ERR%

:: ------------------------------------------------------------
:FIN
endlocal
