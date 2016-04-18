#include <QLayout>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollBar>
#include <QSlider>
#include <QSpinBox>

#include "qtimelinewidget.h"


QTimeLineWidget::QTimeLineWidget(QWidget *parent) : QWidget(parent)
{
    m_spinbox_timeEnd = m_spinbox_timeStart = nullptr;

    QHBoxLayout* layout = new QHBoxLayout(this);
    layout->setMargin(0);

    QHBoxLayout* buttonLayout = new QHBoxLayout(this);
    buttonLayout->setMargin(0);

    QPushButton* playBtn = new QPushButton(QIcon(":/images/play.png"), "", 0);
    QPushButton* nextBtn = new QPushButton(QIcon(":/images/next.png"), "", 0);
    QPushButton* pauseBtn = new QPushButton(QIcon(":/images/pause.png"), "", 0);
    QPushButton* stopBtn = new QPushButton(QIcon(":/images/stop.png"), "", 0);
    QPushButton* rewindBtn = new QPushButton(QIcon(":/images/rewind.png"), "", 0);

    playBtn->setIconSize(QSize(16,16));
    nextBtn->setIconSize(QSize(16,16));
    pauseBtn->setIconSize(QSize(16,16));
    stopBtn->setIconSize(QSize(16,16));
    rewindBtn->setIconSize(QSize(16,16));

    buttonLayout->addWidget(pauseBtn);
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

    m_slider = new QTimeSlider;
    m_slider->setMinimumWidth(300);
    m_slider->setTickInterval( 1 );
    m_slider->setRange(0, 10);
    m_slider->setOrientation(Qt::Horizontal);
    m_slider->setTickPosition(QSlider::TicksBothSides);
//    m_slider->setMinimumHeight(32);

    m_spinbox_timeStart->editingFinished();

    connect(m_spinbox_timeStart, SIGNAL(valueChanged(int)), this, SLOT( setRangeMin(int) ));
    connect(m_spinbox_timeEnd, SIGNAL(valueChanged(int)), this, SLOT( setRangeMax(int) ));

    m_spinbox_timeStart->setValue(0);
    m_spinbox_timeEnd->setValue(20);

    layout->addLayout(buttonLayout);
    layout->addWidget(m_slider);
    layout->addWidget(m_spinbox_timeStart);
    layout->addWidget(m_spinbox_timeEnd);


}

QTimeLineWidget::~QTimeLineWidget()
{

}
