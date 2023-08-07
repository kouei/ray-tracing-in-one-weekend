@echo off
doskey buildd=mkdir .\bin $T CL src\*.cc /EHsc /Fo:bin\ /Fe:bin\main.exe /O2 /Wall /wd4711 /wd4710 /wd5045 /wd4514 /wd4820 /wd4100