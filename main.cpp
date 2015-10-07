#include "mainwindow.h"
#include <QApplication>

#include <QSurfaceFormat>
#include <shaderwindow.h>

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
//  MainWindow w;
//  w.show();

  QSurfaceFormat format;
  format.setSamples(16);

  ShaderWindow window;
  window.setFormat(format);
  window.resize(640,480);
  window.show();

  window.setAnimating(true);

  return a.exec();
}
