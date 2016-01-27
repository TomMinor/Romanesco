#ifndef TESTGLWIDGET_H
#define TESTGLWIDGET_H

#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QMatrix4x4>

#include <QKeyEvent>

#include "optixscene.h"

class TestGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
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

    OptixScene* m_optixScene;

    unsigned int m_previousWidth, m_previousHeight;

private:
    //QMatrix4x4 m_projection;

    int testctr;

    QOpenGLShaderProgram *m_program;

    QVector3D m_camPos, m_desiredCamPos;
    QVector3D m_camRot, m_desiredCamRot;

    GLuint m_vtxPosAttr;
    GLuint m_vtxUVAttr;

    int m_frame;
};

#endif // TESTGLWIDGET_H
