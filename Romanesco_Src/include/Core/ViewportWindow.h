#ifndef SHADERWINDOW_H
#define SHADERWINDOW_H

//#include <SDL.h>
//#include <SDL_haptic.h>
#include <QOpenGLShaderProgram>
#include "OpenGlWindow.h"

#include "OptixScene.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <optix.h>
#include <sutil.h>
#include <optixu/optixpp_namespace.h>

class ShaderWindow : public OpenGLWindow
{
public:
  ShaderWindow();
  ~ShaderWindow();

  virtual void render(QPainter *painter);
  void initialize() Q_DECL_OVERRIDE;
  void render() Q_DECL_OVERRIDE;
  void update() Q_DECL_OVERRIDE;

  OptixScene* m_optixScene;

protected:
  void processMultipleKeyEvents(  );

  //http://stackoverflow.com/questions/7176951/how-to-get-multiple-key-presses-in-single-event
  bool m_firstRelease = true;
  QSet<Qt::Key> keysPressed;
  void keyPressEvent(QKeyEvent* event) Q_DECL_OVERRIDE;
  void keyReleaseEvent(QKeyEvent* event) Q_DECL_OVERRIDE;

  unsigned int m_previousWidth, m_previousHeight;

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
