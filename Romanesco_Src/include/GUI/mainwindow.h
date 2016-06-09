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
#include "highlighter.h"

class QNodeGraph;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void timerEvent(QTimerEvent *_event);

    void showEvent(QShowEvent *_event);

    void keyPressEvent(QKeyEvent* _event);

    void setGlobalStyleSheet(const QString& _styleSheet);

public slots:
    void setRenderPath(std::string _path)
    {
        m_renderPath = _path;
    }

    void setBatchMode(bool _v)
    {
        m_batchMode = _v;
    }

    void loadHitFile(QString _path);

    void loadHitFileDeferred(QString _path)
    {
        m_deferredScenePath = _path;
    }

    void setRender(int _x, int _y)
    {
        m_renderX = _x;
        m_renderY = _y;
    }

    void setStartFrame(int _x)
    {
        m_timeline->setStartFrame(_x);
    }

    void setEndFrame(int _x)
    {
        m_timeline->setEndFrame(_x);
    }

    void setFrameOffset(int _x)
    {
        m_frameOffset = _x;
    }

    void setProgressiveTimeout(int _t)
    {
        m_progressiveTimeout = _t;
        m_progressiveSpinbox->setValue(m_progressiveTimeout);
        m_glViewport->m_optixScene->setProgressiveTimeout(_t);
    }

    void setFOV(float _fov)
    {
        if(!m_fovSpinbox)
        {
            qDebug() << "FOV Spinbox null";
        }

        m_fovSpinbox->setValue( _fov );

//        m_glViewport->setFOV(_fov);
    }

    void setSamples(int _s)
    {
//        m_glViewport->m_optixScene->setSamplesPerPixelSquared(_s);
        m_sqrtNumSamples->setValue(_s);
    }

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

    void updateFrameRefinement(int _frame);
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



    void updateRelativeTime(float _t)
    {
        int currentFrame = m_timeline->getTime();
        unsigned int currentRelativeFrame = currentFrame;
        unsigned int frameRange = m_timeline->getEndFrame() - m_timeline->getStartFrame();

         m_glViewport->updateRelativeTime( currentRelativeFrame );
    }

    void loadHitFile();
    void buildHitFunction();


private:
    void setupEditor()
    {
        QFont font;
        font.setFamily("dejavu-sans-mono");
        font.setFixedPitch(true);
    //    font.setPointSize(10);

        m_editor = new QTextEdit;
        m_editor->setFont(font);
        m_editor->setWordWrapMode( QTextOption::NoWrap );

        m_highlighter = new Highlighter(m_editor->document());

        QWidget* editorWidget = new QWidget;
        QVBoxLayout* layout = new QVBoxLayout;

        QPushButton* loadBtn = new QPushButton;
        loadBtn->setText("Load");
        QPushButton* buildBtn = new QPushButton;
        buildBtn->setText("Build");

        connect(loadBtn, SIGNAL(pressed()), this, SLOT(loadHitFile()));
        connect(buildBtn, SIGNAL(pressed()), this, SLOT(buildHitFunction()));

        QHBoxLayout* buttonLayout = new QHBoxLayout;
        buttonLayout->addWidget(loadBtn);
        buttonLayout->addWidget(buildBtn);

        layout->addLayout(buttonLayout);
        layout->addWidget( m_editor );

        editorWidget->setLayout(layout);

        m_editorTabWidget->insertTab(0, editorWidget, "Scene Hit" );
        m_editorTabWidget->setCurrentIndex(0);

//        QFile file("kernel/tmp.cu");
//        if (file.open(QFile::ReadOnly | QFile::Text))
//            m_editor->setPlainText(file.readAll());
    }

    void setupTabUI();
    void setupSceneSettingsUI();
    void setupMaterialSettingsUI();
    void setupRenderSettingsUI();

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

    QSpinBox* m_sqrtNumSamples;

    TestGLWidget* m_glViewport;
    QAnimatedTimeline* m_timeline;
    QTabWidget* m_mainTabWidget;
    QTabWidget* m_editorTabWidget;

    QWidget* m_sceneSettingsWidget;
    QWidget* m_renderSettingsWidget;
    QWidget* m_materialSettingsWidget;

    bool m_flipbooking;
    bool m_rendering;

    bool m_update_pending;
    bool m_animating;
    int m_updateTimer, m_drawTimer;

    float m_timeScale;

    bool m_batchMode;
    int m_renderX;
    int m_renderY;

    float m_overrideFOV;

    int m_frameOffset;

    Highlighter* m_highlighter;
    QTextEdit* m_editor;

    QDoubleSpinBox* m_fovSpinbox;

    std::string m_renderPath;

    QString m_deferredScenePath;

    int m_progressiveTimeout;
    QSpinBox* m_progressiveSpinbox;
};

#endif // QNEMAINWINDOW_H
