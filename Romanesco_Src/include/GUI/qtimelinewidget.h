#ifndef QTIMELINE_H
#define QTIMELINE_H

#include <QWidget>
#include <QSpinBox>
#include <QSlider>

#include "qtimeslider.h"

class QTimeLineWidget : public QWidget
{
    Q_OBJECT
public:
    explicit QTimeLineWidget(QWidget *parent = 0);
    ~QTimeLineWidget();

signals:
    void play();
    void rewind();
    void stop();
    void nextframe();
    void prevframe();

public slots:
    void setRangeMin(int x)
    {
        if(x > m_spinbox_timeEnd->value())
        {
            x = m_spinbox_timeEnd->value();
            m_spinbox_timeStart->setValue(x);
        }

        if(m_slider)
            m_slider->setMinimum(x);
    }

    void setRangeMax(int x)
    {
        if(x < m_spinbox_timeStart->value())
        {
            x = m_spinbox_timeStart->value();
            m_spinbox_timeEnd->setValue(x);
        }

        if(m_slider)
            m_slider->setMaximum(x);
    }

private:
    QSpinBox* m_spinbox_timeStart;
    QSpinBox* m_spinbox_timeEnd;
    QTimeSlider* m_slider;
};

#endif // QTIMELINE_H
