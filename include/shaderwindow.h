#ifndef SHADERWINDOW_H
#define SHADERWINDOW_H

//#include <SDL.h>
//#include <SDL_haptic.h>
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

  void processMultipleKeyEvents(  );

  //http://stackoverflow.com/questions/7176951/how-to-get-multiple-key-presses-in-single-event
  bool m_firstRelease = true;
  QSet<Qt::Key> keysPressed;
  void keyPressEvent(QKeyEvent* event) Q_DECL_OVERRIDE;
  void keyReleaseEvent(QKeyEvent* event) Q_DECL_OVERRIDE;

private:
  GLuint loadShader(GLenum type, const char *source);

  GLuint m_vtxPosAttr;
  GLuint m_vtxUVAttr;

  GLuint m_resXUniform, m_resYUniform;
  GLuint m_aspectUniform;
  GLuint m_timeUniform;

  GLuint m_posUniform;

  GLuint m_normalMatrix;

  QVector3D m_camPos, m_desiredCamPos;
  QVector3D m_camRot, m_desiredCamRot;

  QOpenGLShaderProgram *m_program;
  int m_frame;

  //SDL_Joystick *js;
};

#endif // SHADERWINDOW_H
