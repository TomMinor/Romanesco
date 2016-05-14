#ifndef QANIMATEDTIMELINE_H
#define QANIMATEDTIMELINE_H

#include <QWidget>
#include <QSpinBox>
#include <QSlider>
#include <QTimeLine>

class QAnimatedTimeline : public QWidget
{
    Q_OBJECT
public:
    explicit QAnimatedTimeline(QWidget *parent = 0);
    virtual ~QAnimatedTimeline() {;}

    int getStartFrame();
    int getEndFrame();

    void setStartFrame(int _x);
    void setEndFrame(int _x);

    void setFPS(int _f);

signals:
    void timeUpdated(float);

public slots:
    void setRangeMin(int x);
    void setRangeMax(int x);
    void setRange(int a, int b);

    void play();
    void rewind();
    void nextFrame();
    void prevFrame();

private slots:
    void updateTime(int _t);
    void updateSlider(int _t);
    void emitTime(int _t);

private:
    QSpinBox* m_spinbox_timeStart;
    QSpinBox* m_spinbox_timeEnd;
    QSlider* m_slider;
    QTimeLine* m_timeline;

    unsigned int m_fps;
};

#endif // QTIMELINE_H
