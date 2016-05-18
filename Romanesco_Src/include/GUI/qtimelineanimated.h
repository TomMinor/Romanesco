#ifndef QANIMATEDTIMELINE_H
#define QANIMATEDTIMELINE_H

#include <QWidget>
#include <QSpinBox>
#include <QSlider>

#include <QTimer>
#include <QTime>

class QAnimatedTimeline : public QWidget
{
    enum Direction
    {
        Forward = 1,
        Backward = -1
    };

    Q_OBJECT
public:
    explicit QAnimatedTimeline(QWidget *parent = 0);
    virtual ~QAnimatedTimeline() {;}

    int getStartFrame();
    int getEndFrame();

    void setStartFrame(int _x);
    void setEndFrame(int _x);

    void setTime(int _f);
    int getTime();
    void setFPS(int _f);

signals:
    void timeUpdated(float);

    void paused();

public slots:
    void setRangeMin(int x);
    void setRangeMax(int x);
    void setRange(int a, int b);


    void play();
    void stop();
    void rewind();
    void gotoNextFrame();
    void gotoPrevFrame();

private slots:
    void updateTime(int _t);
    void updateSlider(int _t);
    void emitTime(int _t);

    void timerUpdate();

    void updateFPS(int _f);


private:
    QSpinBox* m_spinbox_timeStart;
    QSpinBox* m_spinbox_timeEnd;
    QSlider* m_slider;
    QSpinBox* m_currentFrameSpinbox;
    QSpinBox* m_fpsSpinbox;

    QTimer* m_timer;
    QTime m_lastFrameTime;

    Direction m_direction;

    int m_currentFrame;
    unsigned int m_fps;
};

#endif // QTIMELINE_H
