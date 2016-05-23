#include <QScreen>
#include <QDebug>
#include <QKeyEvent>
#include <unistd.h>
#include <math.h>
#include <QtMath>
#include <QDir>

#include <optixu/optixu_math_namespace.h>

#include "ViewportWindow.h"
ShaderWindow::ShaderWindow()
  : m_program(0), m_frame(0), m_optixScene(0), m_previousWidth(0), m_previousHeight(0)
{
    m_desiredCamPos = m_camPos = QVector3D(1.09475, 0.0750364, -1.00239);
    m_desiredCamRot = m_camRot = QVector3D(-0.301546, 0.399876, 0);
}

ShaderWindow::~ShaderWindow()
{
    delete m_optixScene;
}

void ShaderWindow::render(QPainter *painter)
{
    OpenGLWindow::render(painter);
    if( m_previousHeight != height() || m_previousWidth != width() )
    {
        m_previousHeight = height();
        m_previousWidth = width();

        m_optixScene->updateBufferSize( width(), height() );
    }

}

void ShaderWindow::initialize()
{
    QString vertexPath = QDir::currentPath() + "/shaders/raymarch.vert";
    QString fragmentPath = QDir::currentPath() + "/shaders/raymarch.frag";

    QString vertexSrc;
    {
        QFile f(vertexPath);
        if(f.open(QFile::ReadOnly | QFile::Text))
        {
            QTextStream in(&f);
            vertexSrc = in.readAll();
        }
        else
        {
            qDebug() << "Can't open file " << vertexPath;
        }
    }

    QString fragmentSrc;
    {
        QFile f(fragmentPath);
        if(f.open(QFile::ReadOnly | QFile::Text))
        {
            QTextStream in(&f);
            fragmentSrc = in.readAll();
        }
        else
        {
            qDebug() << "Can't open file " << fragmentPath;
        }
    }

    m_program = new QOpenGLShaderProgram(this);
//    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexSrc);
//    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentSrc);
    m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/viewport.vert");
    m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/viewport.frag");

    if(!m_program->link())
    {
        qDebug() << "Link error in shader program\n";
        qDebug() << m_program->log();
        exit(1);
    }

    m_vtxPosAttr = m_program->attributeLocation("vtxPos");
    m_vtxUVAttr = m_program->attributeLocation("vtxUV");
    m_resXUniform = m_program->uniformLocation("resx");
    m_resYUniform = m_program->uniformLocation("resy");
    m_aspectUniform = m_program->uniformLocation("aspect");
    m_timeUniform = m_program->uniformLocation("time");
    m_posUniform = m_program->uniformLocation("pos");
    m_normalMatrix = m_program->uniformLocation("normalMatrix");

    m_optixScene = new OptixScene(width(), height());
}

void ShaderWindow::update()
{

    optix::Matrix4x4 normalmatrix = optix::Matrix4x4::identity();
    optix::Matrix4x4 rotX = optix::Matrix4x4::identity();
    optix::Matrix4x4 rotY = optix::Matrix4x4::identity();
    optix::Matrix4x4 rotZ = optix::Matrix4x4::identity();

    { // X
        float angle = m_camRot.x();
        QVector3D axis = QVector3D(1, 0, 0);
        axis.normalize();
        float s = sin(angle);
        float c = cos(angle);
        float oc = 1.0 - c;


        rotX.setRow(0, optix::make_float4(oc * axis.x() * axis.x() + c,             oc * axis.x() * axis.y() - axis.z() * s,    oc * axis.z() * axis.x() + axis.y() * s,  0.0) );
        rotX.setRow(1, optix::make_float4(oc * axis.x() * axis.y() + axis.z() * s,  oc * axis.y() * axis.y() + c,               oc * axis.y() * axis.z() - axis.x() * s,  0.0) );
        rotX.setRow(2, optix::make_float4(oc * axis.z() * axis.x() - axis.y() * s,  oc * axis.y() * axis.z() + axis.x() * s,    oc * axis.z() * axis.z() + c,             0.0) );
        rotX.setRow(3, optix::make_float4(0.0,                                      0.0,                                        0.0,                                      1.0) );

    }

    { // Y
        float angle = m_camRot.y();
        QVector3D axis = QVector3D(0, 1, 0);
        axis.normalize();
        float s = sin(angle);
        float c = cos(angle);
        float oc = 1.0 - c;

        rotY.setRow(0, optix::make_float4(oc * axis.x() * axis.x() + c,             oc * axis.x() * axis.y() - axis.z() * s,    oc * axis.z() * axis.x() + axis.y() * s,  0.0) );
        rotY.setRow(1, optix::make_float4(oc * axis.x() * axis.y() + axis.z() * s,  oc * axis.y() * axis.y() + c,               oc * axis.y() * axis.z() - axis.x() * s,  0.0) );
        rotY.setRow(2, optix::make_float4(oc * axis.z() * axis.x() - axis.y() * s,  oc * axis.y() * axis.z() + axis.x() * s,    oc * axis.z() * axis.z() + c,             0.0) );
        rotY.setRow(3, optix::make_float4(0.0,                                      0.0,                                        0.0,                                      1.0) );
    }

    { // Z
        float angle = m_camRot.z();
        QVector3D axis = QVector3D(0, 0, 1);
        axis.normalize();
        float s = sin(angle);
        float c = cos(angle);
        float oc = 1.0 - c;

        rotZ.setRow(0, optix::make_float4(oc * axis.x() * axis.x() + c,             oc * axis.x() * axis.y() - axis.z() * s,    oc * axis.z() * axis.x() + axis.y() * s,  0.0) );
        rotZ.setRow(1, optix::make_float4(oc * axis.x() * axis.y() + axis.z() * s,  oc * axis.y() * axis.y() + c,               oc * axis.y() * axis.z() - axis.x() * s,  0.0) );
        rotZ.setRow(2, optix::make_float4(oc * axis.z() * axis.x() - axis.y() * s,  oc * axis.y() * axis.z() + axis.x() * s,    oc * axis.z() * axis.z() + c,             0.0) );
        rotZ.setRow(3, optix::make_float4(0.0,                                      0.0,                                        0.0,                                      1.0) );
    }

    normalmatrix = rotX * rotY * rotZ;



//    m_camPos.setX( FInterpTo( m_camPos.x(), m_desiredCamPos.x(), m_frame, 0.0001) );
//    m_camPos.setY( FInterpTo( m_camPos.y(), m_desiredCamPos.y(), m_frame, 0.0001) );
//    m_camPos.setZ( FInterpTo( m_camPos.z(), m_desiredCamPos.z(), m_frame, 0.0001) );

//    m_camRot.setX( FInterpTo( m_camRot.x(), m_desiredCamRot.x(), m_frame, 0.00025) );
//    m_camRot.setY( FInterpTo( m_camRot.y(), m_desiredCamRot.y(), m_frame, 0.00025) );
//    m_camRot.setZ( FInterpTo( m_camRot.z(), m_desiredCamRot.z(), m_frame, 0.00025) );

    m_optixScene->setVar("global_t", m_frame);

    m_optixScene->setVar("normalmatrix", normalmatrix);
    m_optixScene->setCamera(  optix::make_float3( m_camPos.x(), m_camPos.y(), m_camPos.z() ),
                              90.0f,
                              width(), height()
                              );

  //render();
}

void ShaderWindow::render()
{
  OpenGLWindow::render();

  const qreal retinaScale = devicePixelRatio();
  glViewport(0, 0, width() * retinaScale, height() * retinaScale);

  glClear(GL_COLOR_BUFFER_BIT);

  m_program->bind();

  static GLfloat vertices[] = {
    -1,	-1, 0,
    1,	-1,	0,
    1,	1,	0,
    -1,	-1, 0,
    -1,	1,	0,
    1,	1,	0,
  };
  static GLfloat uv[] = {
      0,	0,
      1,	0,
      1,	1,
      0,	0,
      0,	1,
      1,	1,
  };

  glEnable(GL_TEXTURE_2D);

  glVertexAttribPointer(m_vtxPosAttr, 3, GL_FLOAT, GL_FALSE, 0, vertices);
  glVertexAttribPointer(m_vtxUVAttr, 2, GL_FLOAT, GL_FALSE, 0, uv);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  m_optixScene->drawToBuffer();
  glDrawArrays(GL_TRIANGLES, 0, 6);

  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(0);

  glDisable(GL_TEXTURE_2D);

  m_program->release();

  ++m_frame;
}

GLuint ShaderWindow::loadShader(GLenum type, const char *source)
{
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, 0);
  glCompileShader(shader);
  return shader;
}

void ShaderWindow::keyPressEvent(QKeyEvent* event)
{
  const float offset = 0.025f;
  const float rotateOffset = 0.10f;

  qDebug() << "Arse";

  if( event->key() ==  Qt::Key_A )
  {
      float radius = offset;
      float pitch = m_camRot.x();
      float yaw = m_camRot.y() + ( M_PI/2.0f );

      float pitchRad = pitch;//qDegreesToRadians( pitch );
      float yawRad = yaw;//qDegreesToRadians( yaw );

      //These equations are from the wikipedia page, linked above
      float xMove = radius * -sinf( yawRad ) * cosf( pitchRad );
      float yMove = radius * sinf( pitchRad );
      float zMove = radius * cosf( yawRad ) * cosf( pitchRad );

      m_desiredCamPos.setX( m_desiredCamPos.x() + xMove );
      m_desiredCamPos.setY( m_desiredCamPos.y() + yMove );
      m_desiredCamPos.setZ( m_desiredCamPos.z() + zMove );

  }
  if( event->key() ==  Qt::Key_D )
  {
      float radius = offset;
      float pitch = m_camRot.x();
      float yaw = m_camRot.y() + ( M_PI/2.0f );

      float pitchRad = pitch;//qDegreesToRadians( pitch );
      float yawRad = yaw;//qDegreesToRadians( yaw );

      //These equations are from the wikipedia page, linked above
      float xMove = radius * -sinf( yawRad ) * cosf( pitchRad );
      float yMove = radius * sinf( pitchRad );
      float zMove = radius * cosf( yawRad ) * cosf( pitchRad );

      m_desiredCamPos.setX( m_desiredCamPos.x() - xMove );
      m_desiredCamPos.setY( m_desiredCamPos.y() - yMove );
      m_desiredCamPos.setZ( m_desiredCamPos.z() - zMove );


  }
  if( event->key() ==  Qt::Key_W )
  {
      float radius = offset;
      float pitch = m_camRot.x();
      float yaw = m_camRot.y();

      float pitchRad = pitch;//qDegreesToRadians( pitch );
      float yawRad = yaw;//qDegreesToRadians( yaw );

      //These equations are from the wikipedia page, linked above
      float xMove = radius * -sinf( yawRad ) * cosf( pitchRad );
      float yMove = radius * sinf( pitchRad );
      float zMove = radius * cosf( yawRad ) * cosf( pitchRad );

      m_desiredCamPos.setX( m_desiredCamPos.x() + xMove );
      m_desiredCamPos.setY( m_desiredCamPos.y() + yMove );
      m_desiredCamPos.setZ( m_desiredCamPos.z() + zMove );


  }
  if( event->key() ==  Qt::Key_S )
  {
      float radius = offset;
      float pitch = m_camRot.x();
      float yaw = m_camRot.y();

      float pitchRad = pitch;// qDegreesToRadians( pitch );
      float yawRad = yaw;//qDegreesToRadians( yaw );

      //These equations are from the wikipedia page, linked above
      float xMove = radius * -sinf( yawRad ) * cosf( pitchRad );
      float yMove = radius * sinf( pitchRad );
      float zMove = radius * cosf( yawRad ) * cosf( pitchRad );

      m_desiredCamPos.setX( m_desiredCamPos.x() - xMove );
      m_desiredCamPos.setY( m_desiredCamPos.y() - yMove );
      m_desiredCamPos.setZ( m_desiredCamPos.z() - zMove );
  }
  if( event->key() ==  Qt::Key_Up )
  {
      m_desiredCamRot.setX( m_desiredCamRot.x() + rotateOffset );
  }
  if( event->key() ==  Qt::Key_Down )
  {
      m_desiredCamRot.setX( m_desiredCamRot.x() - rotateOffset );
  }
  if( event->key() ==  Qt::Key_Left )
  {
      m_desiredCamRot.setY( m_desiredCamRot.y() + rotateOffset );
  }
  if( event->key() ==  Qt::Key_Right )
  {
      m_desiredCamRot.setY( m_desiredCamRot.y() - rotateOffset );
  }
  if( event->key() ==  Qt::Key_U )
  {
      qDebug() << "Updating geo";
      m_optixScene->createWorld();
  }
  if( event->key() == Qt::Key_Escape )
    {
      this->close();
    }
}

void ShaderWindow::keyReleaseEvent(QKeyEvent* event)
{
//  if(m_firstRelease)
//  {
//      processMultipleKeyEvents();
//  }
//  m_firstRelease = false;
//  keysPressed.remove((Qt::Key) event->key());
}

void ShaderWindow::processMultipleKeyEvents(  )
{

}
