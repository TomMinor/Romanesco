#include "shaderwindow.h"
#include <QScreen>
#include <QDebug>
#include <unistd.h>
#include <math.h>

ShaderWindow::ShaderWindow()
  : m_program(0), m_frame(0)
{

}

ShaderWindow::~ShaderWindow()
{

}

void ShaderWindow::initialize()
{
  m_program = new QOpenGLShaderProgram(this);
  m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "/home/tom/src/romanesco/Romanesco/raymarch.vert");
  m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "/home/tom/src/romanesco/Romanesco/raymarch.frag");

  if(!m_program->link())
  {
    qDebug() << "Link error in shader program\n";
    exit(1);
  }

  m_vtxPosAttr = m_program->attributeLocation("vtxPos");
  m_vtxUVAttr = m_program->attributeLocation("vtxUV");

  m_center = m_program->uniformLocation("center");
  m_zoom = m_program->uniformLocation("zoom");
  m_c = m_program->uniformLocation("c");

  m_resXUniform = m_program->uniformLocation("resX");
  m_resYUniform = m_program->uniformLocation("resY");
  m_aspectUniform = m_program->uniformLocation("aspect");
  m_timeUniform = m_program->uniformLocation("time");

  //m_matrixUniform = m_program->uniformLocation("matrix");
}

void ShaderWindow::update()
{
  render();
}

void ShaderWindow::render()
{
  OpenGLWindow::render();

  const qreal retinaScale = devicePixelRatio();
  glViewport(0, 0, width() * retinaScale, height() * retinaScale);

  glClear(GL_COLOR_BUFFER_BIT);

  m_program->bind();

  QMatrix4x4 matrix;
  matrix.perspective(60.0f, 4.0f/3.0f, 0.1f, 100.0f);
  matrix.translate(0, 0, -2);
  matrix.rotate(100.0f * m_frame / screen()->refreshRate(), 0, 1, 0);

  QVector2D center(0.0f, 0.0f);
  m_program->setUniformValueArray(m_center, &center, 2);

  QVector2D c(0.0f, 0.0f);
  m_program->setUniformValueArray(m_c, &c, 2);

  QSize res = getResolution();
  float aspect = (res.height() > 0) ? ( (float)res.width() / res.height()) : 1.0f;
  m_program->setUniformValue(m_resXUniform, res.width());
  m_program->setUniformValue(m_resYUniform, res.height());
  m_program->setUniformValue(m_aspectUniform, aspect );

  float time = (float)m_frame / 32.0;
  m_program->setUniformValue(m_timeUniform, time);

  qDebug() << m_frame;

  m_program->setUniformValue(m_zoom, float( ( (sin(m_frame / 32.0) + 1.5 ) * 0.5 ) * 3.0 + 1.0) );
  //m_program->setUniformValue(m_matrixUniform, matrix);

  GLfloat vertices[] = {
    -1,	-1, 0,
    1,	-1,	0,
    1,	1,	0,
    -1,	-1, 0,
    -1,	1,	0,
    1,	1,	0,
  };

  GLfloat uv[] = {
      0,	0,
      1,	0,
      1,	1,
      0,	0,
      0,	1,
      1,	1,
  };

  glVertexAttribPointer(m_vtxPosAttr, 3, GL_FLOAT, GL_FALSE, 0, vertices);
  glVertexAttribPointer(m_vtxUVAttr, 2, GL_FLOAT, GL_FALSE, 0, uv);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  glDrawArrays(GL_TRIANGLES, 0, 6);

  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(0);

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
