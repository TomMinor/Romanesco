#include "OpenGlWindow.h"

#include <QPainter>
#include <QDebug>

OpenGLWindow::OpenGLWindow(QWindow *parent)
  : QWindow(parent),
    m_update_pending(false),
    m_animating(true),
    m_context(0),
    m_device(0)
{
  setSurfaceType(QWindow::OpenGLSurface);

  m_updateTimer = startTimer(30);
  m_drawTimer = startTimer(30);
}

OpenGLWindow::~OpenGLWindow()
{
  killTimer(m_updateTimer);
}


void OpenGLWindow::render(QPainter *painter)
{
  Q_UNUSED(painter);
}

void OpenGLWindow::render()
{
  if (!m_device) {
    m_device = new QOpenGLPaintDevice;
  }

  glEnable(GL_MULTISAMPLE);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  m_device->setSize(size());

  QPainter painter(m_device);
  render(&painter);
}

void OpenGLWindow::initialize()
{

}

void OpenGLWindow::setAnimating(bool animating)
{
  m_animating = animating;

  if (animating) {
    renderLater();
  }
}

void OpenGLWindow::renderLater()
{

}

void OpenGLWindow::renderNow()
{
  if (!isExposed())
    return;

  bool needsInitialize = false;

  if (!m_context) {
    m_context = new QOpenGLContext(this);
    QSurfaceFormat format = requestedFormat();
//    format.setSamples(16);
    m_context->setFormat(format);
    m_context->create();

    needsInitialize = true;
  }

  m_context->makeCurrent(this);

  if (needsInitialize) {
    initializeOpenGLFunctions();
    initialize();
  }

  render();

  m_context->swapBuffers(this);

  if (m_animating)
    renderLater();
}

bool OpenGLWindow::event(QEvent *event)
{
  switch (event->type())
  {
    case QEvent::UpdateRequest:
        m_update_pending = false;
        renderNow();
        return true;
    default:
        return QWindow::event(event);
  }
}

void OpenGLWindow::exposeEvent(QExposeEvent *event)
{
  Q_UNUSED(event);

  if (isExposed())
    renderNow();
}

void OpenGLWindow::update()
{

}

void OpenGLWindow::timerEvent(QTimerEvent *_event)
{
  if(_event->timerId() == m_updateTimer)
  {
    if (isExposed())
    {
      update();
    }
  }
  if(_event->timerId() == m_drawTimer)
  {
    renderNow();
  }
}

void OpenGLWindow::wheelEvent( QWheelEvent *_event)
{
  Q_UNUSED(_event);
}

QSize OpenGLWindow::getResolution() const
{
  if(m_device)
  {
    return m_device->size();
  }
  else
  {
    return QSize(0,0);
  }

}
