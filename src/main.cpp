#include <QApplication>
#include <QSurfaceFormat>

#include <iostream>
#include <vector>
#include <QDebug>

#include "mainwindow.h"
#include "qframebuffer.h"

#include <future>
#include <iostream>

////#include <SDL.h>
////#include <SDL_haptic.h>
//#include <SeExpression.h>

/* TODO
 * Rename TestGLWidget to something better
 * Make all resource paths relative (to what)
 * Why is the viewport so slow (profile?)
 * Cleanup sutil dependency
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

	// Load and set stylesheet
	QFile file("styles/romanesco.qss");
	file.open(QFile::ReadOnly);
	QString stylesheet = QLatin1String(file.readAll());
	window.setGlobalStyleSheet(stylesheet);

	window.setWindowTitle("Romanesco");
	window.resize(1280, 800);
	window.show();

	app.exec();
}
