@echo off
rem SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rem SPDX-License-Identifier: Apache-2.0
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.

rem Note: this only checks that the build and the setup are valid, i.e. that the examples run against
rem the package and produce consistent output. It is not a test of nvJPEG2000 itself.
rem
rem Runs the installed examples on their images against an nvJPEG2000 package and checks
rem what they write, stopping at the first failure. cudart64_<CUDA major>.dll must be on PATH.
rem
rem   run_examples.cmd <nvjpeg2k package root> <CUDA major>

setlocal EnableExtensions

if "%~2"=="" (
    echo Usage: %~nx0 ^<nvjpeg2k-package-root^> ^<cuda-major^>
    exit /b 2
)

for %%I in ("%~dp0.") do set "EXAMPLES_ROOT=%%~fI"
for %%I in ("%~1") do set "PKG_ROOT=%%~fI"
set "CUDA_MAJOR=%~2"
set "LIB_BIN=%PKG_ROOT%\lib\%CUDA_MAJOR%"

if not exist "%LIB_BIN%\nvjpeg2k_0.dll" (
    echo Missing nvJPEG2000 runtime DLL: "%LIB_BIN%\nvjpeg2k_0.dll"
    exit /b 1
)
set "PATH=%LIB_BIN%;%PATH%"

set "BIN=%EXAMPLES_ROOT%\bin"
set "IMAGES=%EXAMPLES_ROOT%\images"
set "OUTPUT=%EXAMPLES_ROOT%\output"

rem Every example and at least one decoder image must be installed, or nothing would be checked.
for %%E in (nvjpeg2k_decode nvjpeg2k_dec_pipelined nvjpeg2k_decode_tile_partial nvjpeg2k_encode) do (
    if not exist "%BIN%\%%E.exe" (
        echo Missing example: "%BIN%\%%E.exe"
        exit /b 1
    )
)
set "FOUND_JP2="
for /r "%IMAGES%\decoder" %%F in (*.jp2) do set "FOUND_JP2=1"
if not defined FOUND_JP2 (
    echo No .jp2 images in "%IMAGES%\decoder"
    exit /b 1
)

rem Start from empty output directories; the samples do not create them.
if exist "%OUTPUT%" rmdir /s /q "%OUTPUT%"
for %%D in (decode pipelined tile_partial encode encode_ht roundtrip roundtrip_ht) do (
    mkdir "%OUTPUT%\%%D" || exit /b 1
)

for /r "%IMAGES%\decoder" %%F in (*.jp2) do (
    call :decode_image "%%F" || exit /b 1
)

rem The encoder defaults to the reversible wavelet, so decoding its output must reproduce the
rem source image exactly, for both block coders.
call :roundtrip "" "" || exit /b 1
call :roundtrip _ht -ht || exit /b 1

echo -- all examples passed
exit /b 0

:decode_image
set "INPUT=%~1"
set "NAME=%~n1"
call :run "%BIN%\nvjpeg2k_decode.exe" -i "%INPUT%" -o "%OUTPUT%\decode" || exit /b 1
call :run "%BIN%\nvjpeg2k_dec_pipelined.exe" -i "%INPUT%" -o "%OUTPUT%\pipelined" || exit /b 1
call :run "%BIN%\nvjpeg2k_decode_tile_partial.exe" -i "%INPUT%" -da 0,0,512,512 -o "%OUTPUT%\tile_partial" || exit /b 1

rem Both decoders make the same library calls, so their outputs must be identical.
call :check_same "%OUTPUT%\decode\%NAME%.bmp" "%OUTPUT%\pipelined\%NAME%.bmp" || exit /b 1
rem Nothing else checks the partial decode, so require its output.
echo -- check "%OUTPUT%\tile_partial\%NAME%.bmp" was written
if not exist "%OUTPUT%\tile_partial\%NAME%.bmp" (
    echo Missing: "%OUTPUT%\tile_partial\%NAME%.bmp"
    exit /b 1
)
exit /b 0

:roundtrip
set "SUFFIX=%~1"
call :run "%BIN%\nvjpeg2k_encode.exe" -i "%IMAGES%\encoder" %~2 -o "%OUTPUT%\encode%SUFFIX%" || exit /b 1
call :run "%BIN%\nvjpeg2k_decode.exe" -i "%OUTPUT%\encode%SUFFIX%" -o "%OUTPUT%\roundtrip%SUFFIX%" || exit /b 1

rem The encoder names its output after its options, so take whatever the decode wrote.
set "DECODED="
set "COUNT=0"
for %%F in ("%OUTPUT%\roundtrip%SUFFIX%\*") do (
    set "DECODED=%%~fF"
    set /a COUNT+=1
)
if not "%COUNT%"=="1" (
    echo Expected one decoded image in "%OUTPUT%\roundtrip%SUFFIX%", found %COUNT%
    exit /b 1
)
call :check_same "%DECODED%" "%IMAGES%\encoder\TestImage640x480.bmp" || exit /b 1
exit /b 0

:run
echo -- %*
%*
exit /b %errorlevel%

:check_same
echo -- check "%~1" matches "%~2"
if not exist "%~1" (
    echo Missing: "%~1"
    exit /b 1
)
if not exist "%~2" (
    echo Missing: "%~2"
    exit /b 1
)
fc /b "%~1" "%~2" >nul || (
    echo "%~1" differs from "%~2"
    exit /b 1
)
exit /b 0
