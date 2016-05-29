#ifndef TESTGLWIDGET_H
#define TESTGLWIDGET_H

#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QMatrix4x4>
#include <QDebug>
#include <QKeyEvent>

#include "OptixScene.h"

class TestGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
public:
    TestGLWidget(QWidget *parent = 0);
    virtual ~TestGLWidget();

    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

    void keyPressEvent(QKeyEvent *_event) override;

    void mouseMoveEvent(QMouseEvent* _event) override;
    void mousePressEvent(QMouseEvent* _event) override;
    void mouseReleaseEvent(QMouseEvent* _event) override;

    void enterEvent(QEvent* _event) override;
    void leaveEvent(QEvent* _event) override;

    void overrideCameraRes(int _width, int _height);

    OptixScene* m_optixScene;
    unsigned int m_previousWidth, m_previousHeight;

public slots:
    void updateTime(float _t)
    {
        m_time = _t;
        m_updateCamera = true;

        m_optixScene->setTime(_t);
    }

signals:
    void initializedGL();

private:
    //QMatrix4x4 m_projection;

    bool m_updateCamera;

    QOpenGLShaderProgram *m_program;

    QVector3D m_camPos, m_desiredCamPos;
    QVector3D m_camRot, m_desiredCamRot;

    GLuint m_vtxPosAttr;
    GLuint m_vtxUVAttr;

    float m_time;
    long m_frame;
};

#endif // TESTGLWIDGET_H
