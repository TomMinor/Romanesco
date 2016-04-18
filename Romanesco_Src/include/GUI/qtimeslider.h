#ifndef QTIMESLIDER_H
#define QTIMESLIDER_H


#include <QSlider>

// http://stackoverflow.com/questions/27531542/tick-marks-disappear-on-styled-qslider
class QTimeSlider : public QSlider
{
    Q_OBJECT

public:
    explicit QTimeSlider(QWidget *parent = 0)
        : QSlider(parent)
    {
        ;
    }
    virtual ~QTimeSlider() {}


protected:
    virtual void paintEvent(QPaintEvent *ev) override;

};

#endif // QTIMESLIDER_H
