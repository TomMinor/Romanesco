/* Copyright (c) 2012, STANISLAW ADASZEWSKI
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of STANISLAW ADASZEWSKI nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL STANISLAW ADASZEWSKI BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

#include <QGraphicsScene>
#include <QFileDialog>
#include <QOpenGLWidget>
#include <QSplitter>

#include "mainwindow.h"

#include "nodegraph/qneblock.h"
#include "nodegraph/qnodeseditor.h"
#include "nodegraph/qneport.h"

#include "nodes/distanceopnode.h"
#include "gridscene.h"

#include "qtimelineanimated.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    QAction *quitAct = new QAction(tr("&Quit"), this);
    quitAct->setShortcuts(QKeySequence::Quit);
    quitAct->setStatusTip(tr("Quit the application"));
    connect(quitAct, SIGNAL(triggered()), qApp, SLOT(quit()));

    QAction *loadAct = new QAction(tr("&Load"), this);
    loadAct->setShortcuts(QKeySequence::Open);
    loadAct->setStatusTip(tr("Open a file"));
    connect(loadAct, SIGNAL(triggered()), this, SLOT(loadFile()));

    QAction *saveAct = new QAction(tr("&Save"), this);
    saveAct->setShortcuts(QKeySequence::Save);
    saveAct->setStatusTip(tr("Save a file"));
    connect(saveAct, SIGNAL(triggered()), this, SLOT(saveFile()));

    QAction *addAct = new QAction(tr("&Add"), this);
    addAct->setStatusTip(tr("Add a block"));
    connect(addAct, SIGNAL(triggered()), this, SLOT(addBlock()));

    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(addAct);
    fileMenu->addAction(loadAct);
    fileMenu->addAction(saveAct);
    fileMenu->addSeparator();
    fileMenu->addAction(quitAct);

    setWindowTitle(tr("Node Editor"));

    QWidget* window = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(window);
//    layout->setMargin(0);

    QSplitter* splitter = new QSplitter(window);
    // Don't try and resize the gl widgets until user
    splitter->setOpaqueResize(false);

    scene = new GridScene(-1000, -1000, 2000, 2000);

    QFormLayout* settingsLayout = new QFormLayout;

//    QVBoxLayout* rhs_layout = new QVBoxLayout;
//    rhs_layout->

//    QWidget* rhs = new QWidget(this);
//    rhs->setLayout(  );

    view = new QGraphicsView(splitter);
    view->setScene(scene);
    //view->setViewport(new QOpenGLWidget);

    view->setRenderHint(QPainter::Antialiasing, true);
    view->setRenderHint(QPainter::HighQualityAntialiasing, true);

    nodeEditor = new QNodeGraph(this);
    nodeEditor->install(scene);

    connect(nodeEditor, SIGNAL(graphChanged()), this, SLOT(graphUpdated()) );

    m_glViewport = 0;
    m_glViewport = new TestGLWidget(splitter);
    m_glViewport->setMinimumWidth(640);
    m_glViewport->setMinimumHeight(480);

    // Add Viewport
    splitter->addWidget(m_glViewport);

    // Add right hand side
    splitter->addWidget(view);

    QAnimatedTimeline* timeline = new QAnimatedTimeline;

    connect(timeline, SIGNAL(timeUpdated(float)), m_glViewport, SLOT(updateTime(float)));

    layout->addWidget(splitter);
    layout->addWidget(timeline);
    window->setLayout(layout);

    setCentralWidget(window);

    m_updateTimer = startTimer(30);
    m_drawTimer = startTimer(30);
}

void MainWindow::timerEvent(QTimerEvent *_event)
{
    if(_event->timerId() == m_updateTimer)
    {
      //if (isExposed())
      if(m_glViewport)
      {
          m_glViewport->update();
      }
    }
}

void MainWindow::graphUpdated()
{
    std::string src = nodeEditor->parseGraph().c_str();
    qDebug() << src.c_str();
    m_glViewport->m_optixScene->createGeometry( src );
}

void MainWindow::timeUpdated(float _t)
{
    qDebug() << "Time : " << _t;
}

MainWindow::~MainWindow()
{
    killTimer(m_drawTimer);
    killTimer(m_updateTimer);
}

void MainWindow::saveFile()
{
	QString fname = QFileDialog::getSaveFileName();
	if (fname.isEmpty())
		return;

	QFile f(fname);
	f.open(QFile::WriteOnly);
	QDataStream ds(&f);
    nodeEditor->save(ds);
}

void MainWindow::loadFile()
{
	QString fname = QFileDialog::getOpenFileName();
	if (fname.isEmpty())
		return;

	QFile f(fname);
	f.open(QFile::ReadOnly);
	QDataStream ds(&f);
    nodeEditor->load(ds);
}

void MainWindow::addBlock()
{
   DistanceOpNode *c = new DistanceOpNode("Union", scene, 0);
//	static const char* names[] = {"Vin", "Voutsadfasdf", "Imin", "Imax", "mul", "add", "sub", "div", "Conv", "FFT"};
//	for (int i = 0; i < 4 + rand() % 3; i++)
//	{
//		b->addPort(names[rand() % 10], rand() % 2, 0, 0);
//        b->setPos(view->sceneRect().center().toPoint());
//	}
}
