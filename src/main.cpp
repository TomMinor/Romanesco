#include <QApplication>
//#include <SDL.h>
//#include <SDL_haptic.h>

#include <SeExpression.h>

#include <QSurfaceFormat>
//#include <shaderwindow.h>

#include <iostream>
#include <vector>
#include <QDebug>

#include "mainwindow.h"

#include "qframebuffer.h"

#include <future>
#include <iostream>



int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  QCoreApplication::setApplicationName("Romanesco Renderer");
  QCoreApplication::setApplicationVersion("1.0");



  QSurfaceFormat format;
  //format.setVersion(4, 3);
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setDepthBufferSize(24);
  format.setStencilBufferSize(8);
  QSurfaceFormat::setDefaultFormat(format);

  MainWindow window;
  //window.setFormat(format);

  // Load and set stylesheet
  QFile file("styles/romanesco.qss");
  file.open(QFile::ReadOnly);
  QString stylesheet = QLatin1String(file.readAll());
  window.setGlobalStyleSheet(stylesheet);

  window.setWindowTitle("Romanesco");
  window.resize(1280, 800);
  window.show();

//  window.setAnimating(true);


  return app.exec();
}
