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

#ifndef QNEMAINWINDOW_H
#define QNEMAINWINDOW_H

#include <QMainWindow>
#include <QtWidgets>

#include "qframebuffer.h"
#include "testglwidget.h"

class QNodeGraph;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void timerEvent(QTimerEvent *_event);

    void keyPressEvent(QKeyEvent* _event);

    void setGlobalStyleSheet(const QString& _styleSheet);


private slots:
	void saveFile();
	void loadFile();
	void addBlock();
    void graphUpdated();
    void timeUpdated(float _t);

    void startFlipbook();
    void cancelFlipbook();

    void startRender();
    void cancelRender();

    void dumpFrame();
    void dumpRenderedFrame();
    void dumpFlipbookFrame();

    void bucketRendered(uint i, uint j);
    void rowRendered(uint _row);

    void initializeGL();

    void setTimeScale(double _f)
    {
        m_timeline->setTimeScale(_f);
    }

    void forceViewportResolution()
    {
        int x = m_resX->value();
        int y = m_resY->value();

//        m_glViewport->overrideCameraRes(x,y);
        m_glViewport->setResolutionOverride( make_int2(x, y) );
    }

private:
    QFramebuffer *m_framebuffer;
    QNodeGraph *nodeEditor;

    QMenu *fileMenu;
    QMenu *renderMenu;
    QAction *m_cancelFlipbookAct;
    QAction *m_cancelRenderAct;
    QAction *m_flipbookAct;
    QAction *m_renderAct;

    QSpinBox* m_resX;
    QSpinBox* m_resY;

    QStatusBar *m_statusBar;
    QProgressBar *m_renderProgress;

    QGraphicsView *view;
    QGraphicsScene *scene;

    TestGLWidget* m_glViewport;
    QAnimatedTimeline* m_timeline;
    QTabWidget* m_mainTabWidget;


    bool m_flipbooking;
    bool m_rendering;

    bool m_update_pending;
    bool m_animating;
    int m_updateTimer, m_drawTimer;

    float m_timeScale;
};

#endif // QNEMAINWINDOW_H
