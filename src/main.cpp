

#include <QApplication>
////#include <SDL.h>
////#include <SDL_haptic.h>
//
//#include <SeExpression.h>

#include <QSurfaceFormat>
////#include <shaderwindow.h>

//#include <OpenImageIO\imageio.h>

#include <iostream>
#include <vector>
#include <QDebug>

#include "mainwindow.h"
#include "qframebuffer.h"

#include <future>
#include <iostream>


/* TODO
 * Fix nvrtc on linux
 * FIx optix crash
 * Rename TestGLWidget to something better
 * Move image code back to OpenEXR
 * Make all resource paths relative (to what)
 * Add nvcc building to cmake
 * Fix texture issue with viewport
 * Why is the viewport so slow (profile?)
 * Cleanup sutil dependency
 * Cleanup OpenEXR dependency
 * Clearly comment nvidia code (PinholeCamera)
 */

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	QCoreApplication::setApplicationName("Romanesco Renderer");
	QCoreApplication::setApplicationVersion("2.0");

	QSurfaceFormat format;
	format.setVersion(4, 3);
	format.setProfile(QSurfaceFormat::CoreProfile);
#ifndef NDEBUG
	format.setOption(QSurfaceFormat::DebugContext);
#endif
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

	optix::Context a = optix::Context::create();


	//  window.setAnimating(true);

	app.exec();
}
