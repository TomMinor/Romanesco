#include "include/GUI/qframebuffer.h"

#include <QLayout>
#include <QMenuBar>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>

QFramebuffer::QFramebuffer(QWidget *parent) : QMainWindow(parent)
{
    setWindowTitle("Flipbook");

    QWidget* widget = new QWidget;
    QVBoxLayout* layout = new QVBoxLayout;
    //    QVBoxLayout* layout = new QVBoxLayout;

    m_menu = new QMenuBar(this);
    this->setMenuBar( m_menu );

    m_timeline = new QAnimatedTimeline;
    m_timeline->setStartFrame(0);
    m_timeline->setEndFrame(0);

    m_view = new QGraphicsView;

    m_scene = new QGraphicsScene;

    setFrame(0);

    m_view->setScene(m_scene);

    layout->addWidget(m_view);
    layout->addWidget(m_timeline);

    widget->setLayout(layout);
    QPalette pal( m_scene->palette() );
    pal.setColor(QPalette::Background, Qt::black);
    m_scene->setPalette(pal);

    this->setCentralWidget(widget);

    this->setMinimumWidth( 40 );
    this->setMinimumHeight( 40 );

    connect(m_timeline, SIGNAL(timeUpdated(float)), this, SLOT(updateFrame(float)));

//    this->setCentralWidget(m_view);
//    this->setLayout( layout );
}

void QFramebuffer::setBufferSize(unsigned int _width, unsigned int _height)
{
    QRectF tmp(0, 0, _width, _height);
    m_scene->setSceneRect(tmp);

    this->setSizePolicy( QSizePolicy::Expanding, QSizePolicy::Expanding );
}

int QFramebuffer::addFrame(const QImage& _frame)
{
    m_frames.push_back( _frame );
    m_timeline->setEndFrame( m_frames.size() - 1);

    setBufferSize(_frame.width(), _frame.height());

    return m_frames.size() - 1;
}

void QFramebuffer::updateFrame(float _f)
{
    setFrame((int)_f);
}

void QFramebuffer::clearFrames()
{
    m_frames.clear();
    m_scene->clear();
    m_timeline->setEndFrame( 0 );
}

void QFramebuffer::setFrame(int _f)
{
    if(_f > m_frames.size() - 1)
    {
        qWarning("Attempted to load frame that doesn't exist (%d)", _f);
        return;
    }

    m_scene->clear();
    m_scene->addPixmap( QPixmap::fromImage( m_frames[_f] ) );
}

