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

//http://doc.qt.io/qt-5/qcommandlineparser.html
enum CommandLineParseResult
{
    CommandLineOk,
    CommandLineError,
    CommandLineVersionRequested,
    CommandLineHelpRequested
};

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, MainWindow *window, QString *errorMessage)
{
//    parser.setSingleDashWordOptionMode(QCommandLineParser::ParseAsLongOptions);

    parser.addHelpOption();
    parser.addVersionOption();
//    parser.addPositionalArgument("source",      QCoreApplication::translate("main", "Source file to copy."));
//    parser.addPositionalArgument("destination", QCoreApplication::translate("main", "Destination directory."));

    const QCommandLineOption startFrameOption("s", "Start frame", "start");
    parser.addOption(startFrameOption);
    const QCommandLineOption endFrameOption("e", "End frame", "end");
    parser.addOption(endFrameOption);

    const QCommandLineOption widthOption("a", "Render width", "width");
    parser.addOption(widthOption);
    const QCommandLineOption heightOption("b", "Render height", "height");
    parser.addOption(heightOption);

//    const QCommandLineOption fovOption("fov", "Field of View", "FOV");
//    parser.addOption(fovOption);

    const QCommandLineOption sceneOption("i", "Scene hit source file to load", "scene");
    parser.addOption(sceneOption);

    const QCommandLineOption outPathOption("f", "Output file path e.g. ./frames/out_%04d.exr", "file");
    parser.addOption(outPathOption);


    parser.addOptions({
                          {{"b", "batch"},
                           QCoreApplication::translate("main", "Quit when the render is complete")},
                      });


//    const QCommandLineOption nameServerOption("n", "The name server to use.", "nameserver");
//    parser.addOption(nameServerOption);
//    const QCommandLineOption typeOption("t", "The lookup type.", "type");
//    parser.addOption(typeOption);
//    parser.addPositionalArgument("name", "The name to look up.");
//    const QCommandLineOption helpOption = parser.addHelpOption();
//    const QCommandLineOption versionOption = parser.addVersionOption();

    if (!parser.parse(QCoreApplication::arguments())) {
        *errorMessage = parser.errorText();
        return CommandLineError;
    }

    if (parser.isSet("v"))
        return CommandLineVersionRequested;

    if (parser.isSet("h"))
        return CommandLineHelpRequested;

    if (parser.isSet("b"))
    {
        window->setBatchMode(true);
    }

    // Frame range
    if (parser.isSet(startFrameOption))
    {
        const QString startFrame = parser.value(startFrameOption);
        window->setStartFrame( startFrame.toInt() );
    }
    if (parser.isSet(endFrameOption))
    {
        const QString endFrame = parser.value(endFrameOption);
        window->setEndFrame( endFrame.toInt() );
    }

    // Render Size
    if (parser.isSet(widthOption) && parser.isSet(heightOption))
    {
        const QString width = parser.value(widthOption);
        const QString height = parser.value(heightOption);
        window->setRender( width.toInt(), height.toInt() );
    }

    if (parser.isSet(sceneOption))
    {
        const QString scenePath = parser.value(sceneOption);
        window->loadHitFileDeferred(scenePath);
    }

    if (parser.isSet(outPathOption))
    {
        const QString outpath = parser.value(outPathOption);
        window->setRenderPath(outpath.toStdString());
    }


//    if (parser.isSet(fovOption))
//    {
//        const QString fov = parser.value(fovOption);
//        window->setFOV(fov.toFloat());
//    }


    return CommandLineOk;
}

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  QCoreApplication::setApplicationName("Romanesco Renderer");
  QCoreApplication::setApplicationVersion("1.0");

  QCommandLineParser parser;
  parser.setApplicationDescription("Fractal Interactive Preview Tool");

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

  QString errorMessage;
  switch (parseCommandLine(parser, &window, &errorMessage)) {
      case CommandLineOk:
          break;
      case CommandLineError:
          fputs(qPrintable(errorMessage), stderr);
          fputs("\n\n", stderr);
          fputs(qPrintable(parser.helpText()), stderr);
          return 1;
      case CommandLineVersionRequested:
          printf("%s %s\n", qPrintable(QCoreApplication::applicationName()),
                 qPrintable(QCoreApplication::applicationVersion()));
          return 0;
      case CommandLineHelpRequested:
          parser.showHelp();
          Q_UNREACHABLE();
  }

  window.setWindowTitle("Romanesco");
  window.resize(1280, 800);
  window.show();

//  window.setAnimating(true);


  return app.exec();
}
