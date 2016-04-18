#ifndef QANIMATEDTIMELINE_H
#define QANIMATEDTIMELINE_H

#include <QWidget>
#include <QSpinBox>
#include <QSlider>
#include <QTimeLine>

#include "qtimeslider.h"

class QAnimatedTimeline : public QWidget
{
    Q_OBJECT
public:
    explicit QAnimatedTimeline(QWidget *parent = 0);
    virtual ~QAnimatedTimeline() {;}

signals:
    void playing();
    void rewinding();
    void stopping();
    void toNextFrame();
    void toPrevFrame();

public slots:
    void setRangeMin(int x);
    void setRangeMax(int x);
    void setRange(int a, int b);

    void play();
    void rewind();
    void nextFrame();
    void prevFrame();

private:
    QSpinBox* m_spinbox_timeStart;
    QSpinBox* m_spinbox_timeEnd;
    QSlider* m_slider;
    QTimeLine* m_timeline;
};

#endif // QTIMELINE_H
