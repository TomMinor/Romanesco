#-------------------------------------------------
#
# Project created by QtCreator 2015-10-06T22:07:49
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Romanesco
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp \
    openglwindow.cpp \
    shaderwindow.cpp

HEADERS  += mainwindow.h \
    openglwindow.h \
    shaderwindow.h

FORMS    += mainwindow.ui

DISTFILES += \
    raymarch.frag \
    raymarch.vert
