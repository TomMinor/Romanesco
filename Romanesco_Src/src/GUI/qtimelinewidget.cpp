#include <QLayout>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollBar>
#include <QSlider>
#include <QSpinBox>
#include <QTimeLine>
#include <QDebug>

#include "qtimelineanimated.h"

QAnimatedTimeline::QAnimatedTimeline(QWidget *parent) : QWidget(parent)
{
    m_spinbox_timeEnd = m_spinbox_timeStart = nullptr;

    QHBoxLayout* layout = new QHBoxLayout(this);
    layout->setMargin(0);

    QHBoxLayout* buttonLayout = new QHBoxLayout(this);
    buttonLayout->setMargin(0);

    QPushButton* playBtn = new QPushButton(QIcon(":/images/play.png"), "", 0);
    QPushButton* nextBtn = new QPushButton(QIcon(":/images/next.png"), "", 0);
    QPushButton* prevBtn = new QPushButton(QIcon(":/images/prev.png"), "", 0);
    QPushButton* stopBtn = new QPushButton(QIcon(":/images/stop.png"), "", 0);
    QPushButton* rewindBtn = new QPushButton(QIcon(":/images/rewind.png"), "", 0);

    m_timeline  = new QTimeLine;
    m_timeline->setLoopCount(99999999);
//    m_timeline->setUpdateInterval( 10.0 );
    m_timeline->setCurveShape(QTimeLine::LinearCurve);

    nextBtn->setIconSize(QSize(16,16));
    rewindBtn->setIconSize(QSize(16,16));
    playBtn->setIconSize(QSize(16,16));
    stopBtn->setIconSize(QSize(16,16));
    prevBtn->setIconSize(QSize(16,16));

    buttonLayout->addWidget(prevBtn);
    buttonLayout->addWidget(rewindBtn);
    buttonLayout->addWidget(stopBtn);
    buttonLayout->addWidget(playBtn);
    buttonLayout->addWidget(nextBtn);

    m_spinbox_timeStart = new QSpinBox;
    m_spinbox_timeStart->setMaximumWidth(50);
    m_spinbox_timeStart->setMaximum(2000);

    m_spinbox_timeEnd = new QSpinBox;
    m_spinbox_timeEnd->setMaximumWidth(50);
    m_spinbox_timeEnd->setMaximum(2000);

    m_slider = new QSlider;
    m_slider->setMinimumWidth(300);
    m_slider->setTickInterval( 1 );
    m_slider->setRange(0, 10);
    m_slider->setOrientation(Qt::Horizontal);
    m_slider->setTickPosition(QSlider::TicksBothSides);
//    m_slider->setMinimumHeight(32);

    connect(playBtn, SIGNAL(pressed()), this, SLOT(play()));
    connect(stopBtn, SIGNAL(pressed()), m_timeline, SLOT(stop()));
    connect(rewindBtn, SIGNAL(pressed()), this, SLOT(rewind()));
    connect(prevBtn, SIGNAL(pressed()), this, SLOT(prevFrame()));
    connect(nextBtn, SIGNAL(pressed()), this, SLOT(nextFrame()));

    connect(m_slider, SIGNAL(rangeChanged(int,int)), this, SLOT(setRange(int,int)) );
    connect(m_timeline, SIGNAL(frameChanged(int)), m_slider, SLOT(setValue(int)));
    connect(m_spinbox_timeStart, SIGNAL(valueChanged(int)), this, SLOT( setRangeMin(int) ));
    connect(m_spinbox_timeEnd, SIGNAL(valueChanged(int)), this, SLOT( setRangeMax(int) ));

    m_spinbox_timeStart->setValue(0);
    m_spinbox_timeEnd->setValue(200);

    layout->addLayout(buttonLayout);
    layout->addWidget(m_slider);
    layout->addWidget(m_spinbox_timeStart);
    layout->addWidget(m_spinbox_timeEnd);


}

void QAnimatedTimeline::play()
{
    m_timeline->setDirection(QTimeLine::Forward);
    m_timeline->resume();
}

void QAnimatedTimeline::rewind()
{
    m_timeline->setDirection(QTimeLine::Backward);
    m_timeline->start();
}

void QAnimatedTimeline::nextFrame()
{
    m_slider->setValue(m_slider->value() + 1);
}

void QAnimatedTimeline::prevFrame()
{
    m_slider->setValue(m_slider->value() - 1);
}

void QAnimatedTimeline::setRangeMin(int x)
{
    if(x > m_spinbox_timeEnd->value())
    {
        x = m_spinbox_timeEnd->value();
        m_spinbox_timeStart->setValue(x);
    }

    if(m_slider)
        m_slider->setMinimum(x);
}

void QAnimatedTimeline::setRangeMax(int x)
{
    if(x < m_spinbox_timeStart->value())
    {
        x = m_spinbox_timeStart->value();
        m_spinbox_timeEnd->setValue(x);
    }

    if(m_slider)
        m_slider->setMaximum(x);
}

void QAnimatedTimeline::setRange(int a, int b)
{
    m_timeline->setFrameRange(a,b);
}
