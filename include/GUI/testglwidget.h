#ifndef TESTGLWIDGET_H
#define TESTGLWIDGET_H

#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QMatrix4x4>
#include <QDebug>
#include <QKeyEvent>
#include <QTimer>

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

//    void overrideCameraRes(int _width, int _height);

    OptixScene* m_optixScene;
    unsigned int m_previousWidth, m_previousHeight;


    bool getResolutionOverride()
    {
        return m_overrideRes;
    }

public slots:
    void updateTime(float _t)
    {
        m_time = _t;
        m_updateCamera = true;

        m_optixScene->setTime(_t);
    }

    void updateRelativeTime(float _t)
    {
        m_updateCamera = true;

        m_optixScene->setRelativeTime(_t);
    }

    void setShouldOverrideResolution(bool _v)
    {
        m_overrideRes = _v;
        m_updateCamera = true;
    }

    void setResolutionOverride(int2 _res)
    {
        m_overrideWidth = _res.x;
        m_overrideHeight = _res.y;

        m_optixScene->updateBufferSize( m_overrideWidth, m_overrideHeight );
        updateCamera();
        m_updateCamera = true;
    }

    void setFOV(double _fov)
    {
        m_fov = _fov;
        updateCamera();
        m_updateCamera = true;
    }

    float getFOV()
    {
        return m_fov;
    }

    void setCameraPos(float _x, float _y, float _z)
    {
        m_camPos = m_desiredCamPos = QVector3D(_x, _y, _z);
        updateCamera();
    }

    void setCameraRot(float _x, float _y, float _z)
    {
        m_camRot = m_desiredCamRot = QVector3D(_x, _y, _z);
        updateCamera();
    }

    void refreshScene()
    {
        m_updateCamera = true;
    }

    void updateScene();

signals:
    void initializedGL();

private:
    //QMatrix4x4 m_projection;

    void updateCamera();

    bool m_updateCamera;

    QOpenGLShaderProgram *m_program;

    QVector3D m_camPos, m_desiredCamPos;
    QVector3D m_camRot, m_desiredCamRot;

    GLuint m_vtxPosAttr;
    GLuint m_vtxUVAttr;

    float m_time;
    long m_frame;
    float m_fov;

    bool m_overrideRes;
    unsigned int m_overrideWidth;
    unsigned int m_overrideHeight;

    QTimer* m_timer;
};

#endif // TESTGLWIDGET_H
