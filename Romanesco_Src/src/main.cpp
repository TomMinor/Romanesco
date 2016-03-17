#include "mainwindow.h"
#include <QApplication>
//#include <SDL.h>
//#include <SDL_haptic.h>

#include <QSurfaceFormat>
//#include <shaderwindow.h>

#include <mainwindow.h>

#include <iostream>
#include <vector>
#include <QDebug>

int main(int argc, char *argv[])
{
//  if (SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC) < 0 )
//  {
//    // Or die on error
//    qCritical("Unable to initialize SDL");
//  }

//  int numJoyPads = SDL_NumJoysticks();
//  if(numJoyPads ==0) {
//    qWarning( "No joypads found" );
//  } else {
//    qDebug( "Found %d joypads", numJoyPads );
//  }

  QApplication a(argc, argv);
//  MainWindow w;
//  w.show();

  QSurfaceFormat format;
  //format.setVersion(4, 3);
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setDepthBufferSize(24);
  format.setStencilBufferSize(8);
  QSurfaceFormat::setDefaultFormat(format);

  //haderWindow window;
  MainWindow window;
  //window.setFormat(format);
  window.resize(800, 600);
  window.show();

  //window.setAnimating(true);

  return a.exec();
}
