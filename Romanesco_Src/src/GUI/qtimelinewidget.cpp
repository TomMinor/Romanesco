#include <QLayout>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollBar>
#include <QSlider>
#include <QSpinBox>
#include <QTimeLine>
#include <QDebug>

#include <assert.h>

#include "qtimelineanimated.h"

QAnimatedTimeline::QAnimatedTimeline(QWidget *parent) : QWidget(parent)
{
    m_direction = Direction::Forward;

    // Setup UI
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

    // Set this up like the Fabric guys did https://github.com/fabric-engine/FabricUI/blob/arbus/Viewports/TimeLineWidget.cpp
    // QTimer is inaccurate, so set the interval to be as rapid as possible and figure out the update rate ourselves
    m_timer = new QTimer(this);
    m_timer->setInterval(3);

    connect(m_timer, SIGNAL(timeout()), this, SLOT(timerUpdate()));

    m_timeline  = new QTimeLine;
    m_timeline->setLoopCount(99999999);
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
    m_spinbox_timeStart->setMaximumWidth(150);
    m_spinbox_timeStart->setMaximum(200000);

    m_spinbox_timeEnd = new QSpinBox;
    m_spinbox_timeEnd->setMaximumWidth(150);
    m_spinbox_timeEnd->setMaximum(200000);

    m_slider = new QSlider;
    m_slider->setMinimumWidth(300);
    m_slider->setTickInterval( 10 );
    m_slider->setRange(0, 10);
    m_slider->setOrientation(Qt::Horizontal);
    m_slider->setTickPosition(QSlider::TicksAbove);

    connect(playBtn, SIGNAL(pressed()), this, SLOT(play()));
    connect(stopBtn, SIGNAL(pressed()), m_timeline, SLOT(stop()));
    connect(rewindBtn, SIGNAL(pressed()), this, SLOT(rewind()));
    connect(prevBtn, SIGNAL(pressed()), this, SLOT(prevFrame()));
    connect(nextBtn, SIGNAL(pressed()), this, SLOT(nextFrame()));

    ///@todo Fix draggable time slider
    connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(updateTime(int)));
//    connect(m_timeline, SIGNAL(frameChanged(int)), this, SLOT(updateSlider(int)));

    connect(m_timeline, SIGNAL(frameChanged(int)), this, SLOT(emitTime(int)));

    connect(m_spinbox_timeStart, SIGNAL(valueChanged(int)), this, SLOT( setRangeMin(int) ));
    connect(m_spinbox_timeEnd, SIGNAL(valueChanged(int)), this, SLOT( setRangeMax(int) ));

    m_spinbox_timeStart->setValue(0);
    m_spinbox_timeEnd->setValue(250);

    layout->addLayout(buttonLayout);
    layout->addWidget(m_slider);
    layout->addWidget(m_spinbox_timeStart);
    layout->addWidget(m_spinbox_timeEnd);

    setFPS(10);
}

void QAnimatedTimeline::timerUpdate()
{
    // We will be getting about 1 call per milli-second,
    // however QTimer is really not precise so we cannot rely
    // on its delay.
    double ms = m_lastFrameTime.elapsed();
    if( m_fps > 0 && ms + 0.5 < 1000.0 / m_fps ) // Add 0.5 so we have a better average framerate (else we are always above)
    {
        return; // Wait longer
    }

    m_lastFrameTime.start();

//    int newtime =

}

void QAnimatedTimeline::setFPS(int _f)
{
    m_fps = _f;
    const float fps_ms = (1.0f / m_fps) * 1000.0f;
    if(m_timeline)
    {
        m_timeline->setUpdateInterval( fps_ms );
    }
}

void QAnimatedTimeline::setStartFrame(int _x)
{
    m_spinbox_timeStart->setValue(_x);
}

void QAnimatedTimeline::setEndFrame(int _x)
{
    m_spinbox_timeEnd->setValue(_x);
}

void QAnimatedTimeline::updateSlider(int _x)
{
    qDebug() << "Updating slider";
    m_slider->setValue(_x / m_fps);
}

void QAnimatedTimeline::updateTime(int _x)
{
    qDebug() << "Updating time"  << _x / m_fps;
    m_timeline->setCurrentTime( _x );
}

void QAnimatedTimeline::emitTime(int _x)
{
    qDebug() << m_timeline->frameForTime( m_timeline->currentTime() ) / m_fps;// / m_fps;
    m_slider->setValue(_x / m_fps);
    emit timeUpdated( _x / m_fps );
}

int QAnimatedTimeline::getStartFrame()
{
    if(!m_timeline)
    {
        assert(0 && "Timeline pointer invalid");
    }
    return m_timeline->startFrame();
}

int QAnimatedTimeline::getEndFrame()
{
    if(!m_timeline)
    {
        assert(0 && "Timeline pointer invalid");
    }
    return m_timeline->endFrame();
}

void QAnimatedTimeline::play()
{
    m_timeline->setDirection(QTimeLine::Forward);

    if(m_timeline->state() ==  QTimeLine::Running )
    {
        m_timeline->resume();
    } else {
        m_timeline->start();
    }
}

void QAnimatedTimeline::rewind()
{
    m_timeline->setDirection(QTimeLine::Backward);
    m_timeline->start();
}

void QAnimatedTimeline::nextFrame()
{
//    m_slider->setValue(m_slider->value() + 1);
//    m_timeline->setCurrentTime( m_timeline->currentFrame()+ 1 );
//    qDebug() << m_timeline->valueForTime( m_timeline->currentFrame()+ 1);

}

void QAnimatedTimeline::prevFrame()
{
//    m_slider->setValue(m_slider->value() - 1);
//    m_timeline->setCurrentTime( m_timeline->valueForTime( m_slider->value() - 1) );
}

void QAnimatedTimeline::setRangeMin(int x)
{
    if(x > m_spinbox_timeEnd->value())
    {
        x = m_spinbox_timeEnd->value();
        m_spinbox_timeStart->setValue(x);
    }

    if(m_slider)
    {
        m_slider->setMinimum(x * m_fps);
    }

    m_timeline->setStartFrame(x * m_fps);

    unsigned int difference = m_timeline->endFrame() - m_timeline->startFrame();
    m_timeline->setDuration( difference + 1);
}

void QAnimatedTimeline::setRangeMax(int x)
{
    if(x < m_spinbox_timeStart->value())
    {
        x = m_spinbox_timeStart->value();
        m_spinbox_timeEnd->setValue(x);
    }

    if(m_slider)
    {
        m_slider->setMaximum(x * m_fps);
    }

    m_timeline->setEndFrame(x * m_fps);

    unsigned int difference = m_timeline->endFrame() - m_timeline->startFrame();
    m_timeline->setDuration( difference + 1);
}

void QAnimatedTimeline::setRange(int a, int b)
{
    m_timeline->setFrameRange(a,b);
}
