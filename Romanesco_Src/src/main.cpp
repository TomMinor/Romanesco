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


int twice(int m) {
  return 2 * m;
}

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  QCoreApplication::setApplicationName("Romanesco Renderer");
  QCoreApplication::setApplicationVersion("1.0");

  QCommandLineParser parser;
  parser.setApplicationDescription("Test helper");
  parser.addHelpOption();
  parser.addVersionOption();
  parser.addPositionalArgument("source",      QCoreApplication::translate("main", "Source file to copy."));
  parser.addPositionalArgument("destination", QCoreApplication::translate("main", "Destination directory."));

  parser.addOptions({
                        {"p",
                         QCoreApplication::translate("main", "Show progress during copy")},

                        {{"f", "force"},
                         QCoreApplication::translate("main", "Overwrite existing files.")},

                        {{"t", "target-directory"},
                         QCoreApplication::translate("main", "Copy all source files into <directory>."),
                         QCoreApplication::translate("main", "directory")},
                    });

  parser.process(app);

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
