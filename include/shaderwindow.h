#ifndef SHADERWINDOW_H
#define SHADERWINDOW_H

#include <QOpenGLShaderProgram>
#include "openglwindow.h"

class ShaderWindow : public OpenGLWindow
{
public:
  ShaderWindow();
  ~ShaderWindow();

  void initialize() Q_DECL_OVERRIDE;
  void render() Q_DECL_OVERRIDE;
  void update() Q_DECL_OVERRIDE;

protected:

  void keyPressEvent(QKeyEvent* event) Q_DECL_OVERRIDE;

private:
  GLuint loadShader(GLenum type, const char *source);

  GLuint m_vtxPosAttr;
  GLuint m_vtxUVAttr;

  GLuint m_rotateMatrixUniform, m_transMatrixUniform;

  GLuint m_resXUniform, m_resYUniform;
  GLuint m_aspectUniform;
  GLuint m_timeUniform;

  GLuint m_pitchUniform, m_yawUniform, m_posUniform, m_rotUniform;

  GLuint m_matrix, m_normalMatrix;

  GLuint m_center;
  GLuint m_zoom;
  GLuint m_c;

  QVector3D m_camPos;
  QVector3D m_camRot;

  QOpenGLShaderProgram *m_program;
  int m_frame;
};

#endif // SHADERWINDOW_H
