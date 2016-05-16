#ifndef QFRAMEBUFFER_H
#define QFRAMEBUFFER_H


#include <QMainWindow>
#include <QGraphicsScene>
#include <QImage>
#include <QList>

#include "GUI/qtimelineanimated.h"

class QFramebuffer : public QMainWindow
{
    Q_OBJECT
public:
    explicit QFramebuffer(QWidget *parent = 0);

    int addFrame(const QImage& _frame);
    void clearFrames();
    void setFrame(int _f);

    void setBufferSize(unsigned int _width, unsigned int _height);

private:
    QAnimatedTimeline* m_timeline;
    QGraphicsScene* m_scene;
    QMenuBar* m_menu;
    QGraphicsView* m_view;

    QList<QImage> m_frames;

signals:


public slots:
    void updateFrame(float _f);
};

#endif // QFRAMEBUFFER_H
