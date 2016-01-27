#ifndef OPENGLWINDOW_H
#define OPENGLWINDOW_H

#include <QWindow>
#include <QOpenGLFunctions>
#include <QOpenGLPaintDevice>

class OpenGLWindow : public QWindow, protected QOpenGLFunctions
{
  Q_OBJECT

public:
  explicit OpenGLWindow(QWindow *parent = 0);
  ~OpenGLWindow();

  virtual void render(QPainter *painter);
  virtual void render();
  virtual void update();

  virtual void initialize();

  QSize getResolution() const;

  void setAnimating(bool animating);

public slots:
  void renderLater();
  void renderNow();

protected:
  bool event(QEvent *event) Q_DECL_OVERRIDE;

  void exposeEvent(QExposeEvent *event) Q_DECL_OVERRIDE;

  void timerEvent(QTimerEvent *_event) Q_DECL_OVERRIDE;
  void wheelEvent( QWheelEvent *_event) Q_DECL_OVERRIDE;

private:
    bool m_update_pending;
    bool m_animating;
    int m_updateTimer, m_drawTimer;

    QOpenGLContext *m_context;
    QOpenGLPaintDevice *m_device;
};

#endif // OPENGLWINDOW_H
