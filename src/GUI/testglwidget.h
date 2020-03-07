#ifndef TESTGLWIDGET_H
#define TESTGLWIDGET_H

#include <QOpenGLShaderProgram>
//#include <QOpenGLFunctions>
#include <QOpenGLFunctions_4_3_Core>
//#include <qopenglfunctions_4_3_compatibility.h>
#include <QOpenGLDebugMessage>
#include <QOpenGLDebugLogger>
#include <QOpenGLWidget>
#include <QMatrix4x4>
#include <QDebug>
#include <QKeyEvent>
#include <QTimer>

#include <glm/vec2.hpp>

//#include "OptixScene.h"


class QOpenGLDebugMessage;
class QOpenGLDebugLogger;

class TestGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_4_3_Core
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

	///@todo This should be private
    //OptixScene* m_optixScene;
    unsigned int m_previousWidth, m_previousHeight;


    bool getResolutionOverride()
    {
        return m_overrideRes;
    }

protected slots:
	void messageLogged(const QOpenGLDebugMessage &msg);

public slots:
    void updateTime(float _t)
    {
        m_time = _t;
        m_updateCamera = true;

		//if (m_optixScene)
		//    m_optixScene->setTime(_t);
    }

    void updateRelativeTime(float _t)
    {
        m_updateCamera = true;

		//if (m_optixScene)
		//	m_optixScene->setRelativeTime(_t);
    }

    void setShouldOverrideResolution(bool _v)
    {
        m_overrideRes = _v;
        m_updateCamera = true;
    }

	void setResolutionOverride(optix::int2 _res)
    {
        m_overrideWidth = _res.x;
        m_overrideHeight = _res.y;

		//if (m_optixScene)
		//	m_optixScene->updateBufferSize( m_overrideWidth, m_overrideHeight );
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
	void printGLErrors();

private:
    //QMatrix4x4 m_projection;

    void updateCamera();

    bool m_updateCamera;

    QOpenGLShaderProgram *m_program;
	QOpenGLDebugLogger *m_debugLogger;

    QVector3D m_camPos, m_desiredCamPos;
    QVector3D m_camRot, m_desiredCamRot;

    GLint m_vtxPosAttr;
    GLint m_vtxUVAttr;
	GLuint m_vao;
	GLuint m_vboPos;
	GLuint m_vboUV;

    float m_time;
    long m_frame;
    float m_fov;

    bool m_overrideRes;
    unsigned int m_overrideWidth;
    unsigned int m_overrideHeight;

    QTimer* m_timer;
};

#endif // TESTGLWIDGET_H
