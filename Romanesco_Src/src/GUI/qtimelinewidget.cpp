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

    setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed));
    setContentsMargins(0, 0, 0, 0);


    // Set this up like the Fabric guys did https://github.com/fabric-engine/FabricUI/blob/arbus/Viewports/TimeLineWidget.cpp
    // QTimer is inaccurate, so set the interval to be as rapid as possible and figure out the update rate ourselves
    m_timer = new QTimer(this);
    m_timer->setInterval(3);

    connect(m_timer, SIGNAL(timeout()), this, SLOT(timerUpdate()));

    m_slider = new QSlider;
    m_slider->setMinimumWidth(300);
    m_slider->setTickInterval( 10 );
    m_slider->setRange(0, 10);
    m_slider->setOrientation(Qt::Horizontal);
    m_slider->setTickPosition(QSlider::TicksBelow);

    QHBoxLayout* buttonLayout = new QHBoxLayout(this);
    buttonLayout->setMargin(0);

    QPushButton* playBtn = new QPushButton(QIcon(":/images/play.png"), "", 0);
    QPushButton* nextBtn = new QPushButton(QIcon(":/images/next.png"), "", 0);
    QPushButton* prevBtn = new QPushButton(QIcon(":/images/prev.png"), "", 0);
    QPushButton* stopBtn = new QPushButton(QIcon(":/images/stop.png"), "", 0);
    QPushButton* rewindBtn = new QPushButton(QIcon(":/images/rewind.png"), "", 0);

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

    QHBoxLayout* spinnerLayout = new QHBoxLayout(this);
    spinnerLayout->setMargin(0);

    m_spinbox_timeStart = new QSpinBox;
    m_spinbox_timeStart->setMaximumWidth(150);
    m_spinbox_timeStart->setMaximum(1000000);
    m_spinbox_timeStart->setMinimum(-1000000);
    m_spinbox_timeStart->setFrame(false);
    m_spinbox_timeStart->setWrapping(false);
    m_spinbox_timeStart->setAlignment(Qt::AlignCenter|Qt::AlignTrailing|Qt::AlignVCenter);

    m_spinbox_timeEnd = new QSpinBox;
    m_spinbox_timeEnd->setMaximumWidth(150);
    m_spinbox_timeEnd->setMaximum(1000000);
    m_spinbox_timeEnd->setMinimum(-1000000);
    m_spinbox_timeEnd->setFrame(false);
    m_spinbox_timeEnd->setWrapping(false);
    m_spinbox_timeEnd->setAlignment(Qt::AlignCenter|Qt::AlignTrailing|Qt::AlignVCenter);

    m_currentFrameSpinbox = new QSpinBox;
    m_currentFrameSpinbox->setMaximumWidth(150);
    m_currentFrameSpinbox->setMaximum(1000000);
    m_currentFrameSpinbox->setMinimum(-1000000);
    m_currentFrameSpinbox->setFrame(false);
    m_currentFrameSpinbox->setWrapping(false);
    m_currentFrameSpinbox->setAlignment(Qt::AlignCenter|Qt::AlignTrailing|Qt::AlignVCenter);
    m_currentFrameSpinbox->setButtonSymbols(QAbstractSpinBox::NoButtons);

    connect(m_slider, SIGNAL(valueChanged(int)), m_currentFrameSpinbox, SLOT(setValue(int)));

    m_fpsSpinbox = new QSpinBox;
    m_fpsSpinbox->setMinimumWidth(40);
    m_fpsSpinbox->setMaximumWidth(40);
    m_fpsSpinbox->setMaximum(120);
    m_fpsSpinbox->setMinimum(1);
    m_fpsSpinbox->setFrame(false);
    m_fpsSpinbox->setWrapping(false);
    m_fpsSpinbox->setAlignment(Qt::AlignCenter|Qt::AlignTrailing|Qt::AlignVCenter);
    m_fpsSpinbox->setButtonSymbols(QAbstractSpinBox::NoButtons);

    connect(m_fpsSpinbox, SIGNAL(valueChanged(int)), this, SLOT(updateFPS(int)));

    spinnerLayout->addWidget(m_currentFrameSpinbox);
    spinnerLayout->addWidget(m_spinbox_timeStart);
    spinnerLayout->addWidget(m_spinbox_timeEnd);
    spinnerLayout->addWidget(m_fpsSpinbox);

    connect(playBtn, SIGNAL(pressed()), this, SLOT(play()));
    connect(stopBtn, SIGNAL(pressed()), this, SLOT(stop()));
    connect(rewindBtn, SIGNAL(pressed()), this, SLOT(rewind()));
    connect(prevBtn, SIGNAL(pressed()), this, SLOT(gotoPrevFrame()));
    connect(nextBtn, SIGNAL(pressed()), this, SLOT(gotoNextFrame()));

    connect(m_currentFrameSpinbox, SIGNAL(valueChanged(int)), m_slider, SLOT(setValue(int)));

    connect(m_currentFrameSpinbox, SIGNAL(valueChanged(int)), this, SLOT(updateTime(int)));


    connect(m_spinbox_timeStart, SIGNAL(valueChanged(int)), this, SLOT( setRangeMin(int) ));
    connect(m_spinbox_timeEnd, SIGNAL(valueChanged(int)), this, SLOT( setRangeMax(int) ));

    m_spinbox_timeStart->setValue(0);
    m_spinbox_timeEnd->setValue(250);

    layout->addLayout(buttonLayout);
    layout->addWidget(m_slider);
    layout->addLayout(spinnerLayout);

    setFPS(30);
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

    int newtime = getTime() + m_direction;

    if(newtime > getEndFrame())
    {
        newtime = getStartFrame();
    }
    else if(newtime < getStartFrame())
    {
        newtime = getEndFrame();
    }

    setTime(newtime);
}

void QAnimatedTimeline::setTime(int _f)
{
    m_slider->setValue(_f);
    m_currentFrameSpinbox->setValue(_f);
}

int QAnimatedTimeline::getTime()
{
    return static_cast<int>(m_currentFrameSpinbox->value());
}

void QAnimatedTimeline::setFPS(int _f)
{
    m_fpsSpinbox->setValue(_f);
}

void QAnimatedTimeline::updateFPS(int _f)
{
    m_fps = _f;
    const float fps_ms = (1.0f / m_fps) * 1000.0f;
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

}

void QAnimatedTimeline::updateTime(int _x)
{
    emit timeUpdated(_x);
}

void QAnimatedTimeline::emitTime(int _x)
{

}

int QAnimatedTimeline::getStartFrame()
{
    if(!m_spinbox_timeStart)
    {
        assert(0 && "Spinbox start pointer invalid");
    }
    return m_spinbox_timeStart->value();
}

int QAnimatedTimeline::getEndFrame()
{
    if(!m_spinbox_timeEnd)
    {
        assert(0 && "Spinbox end pointer invalid");
    }
    return m_spinbox_timeEnd->value();
}

void QAnimatedTimeline::play()
{
    m_direction = Direction::Forward;
    m_timer->start();
    m_lastFrameTime.start();
}

void QAnimatedTimeline::stop()
{
    m_timer->stop();
    emit paused();
}

void QAnimatedTimeline::rewind()
{
    m_direction = Direction::Backward;
    m_timer->start();
    m_lastFrameTime.start();
}

void QAnimatedTimeline::gotoNextFrame()
{
    int frame = getTime();
    setTime( frame + 1);
}

void QAnimatedTimeline::gotoPrevFrame()
{
    int frame = getTime();
    setTime( frame - 1);
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
        m_slider->setMinimum(x);
    }

    m_currentFrameSpinbox->setMinimum(x);

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
        m_slider->setMaximum(x);
    }

    m_currentFrameSpinbox->setMaximum(x);
}

void QAnimatedTimeline::setRange(int a, int b)
{

}
