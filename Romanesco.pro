#-------------------------------------------------
#
# Project created by QtCreator 2015-10-06T22:07:49
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Romanesco
TEMPLATE = app

SOURCES += src/*.cpp

HEADERS  += include/*.h

INCLUDEPATH += include/

OBJECTS_DIR = .obj/
MOC_DIR = .moc/
UI_DIR = .ui/
RCC_DIR = .rcc/

FORMS    += ui/mainwindow.ui

DISTFILES += \
    shaders/raymarch.frag \
    shaders/raymarch.vert
