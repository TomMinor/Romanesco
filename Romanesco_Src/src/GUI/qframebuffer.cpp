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
    m_timeline->setEndFrame(3);

    addFrame( QImage("/home/tom/test/1.png") );
    addFrame( QImage("/home/tom/test/2.png") );
    addFrame( QImage("/home/tom/test/3.png") );
    addFrame( QImage("/home/tom/test/4.png") );
    addFrame( QImage("/home/tom/test/5.png") );

    QImage& image = m_frames[0];

    m_view = new QGraphicsView;

    m_scene = new QGraphicsScene;

    QRectF tmp(0, 0, image.width(), image.height());
    m_scene->setSceneRect(tmp);

    setFrame(0);

    m_view->setScene(m_scene);

    layout->addWidget(m_view);
    layout->addWidget(m_timeline);

    widget->setLayout(layout);

    this->setCentralWidget(widget);

    connect(m_timeline, SIGNAL(timeUpdated(float)), this, SLOT(updateFrame(float)));

//    this->setCentralWidget(m_view);
//    this->setLayout( layout );
}

void QFramebuffer::addFrame(const QImage& _frame)
{
    m_frames.push_back( _frame );
    m_timeline->setEndFrame( m_frames.size() - 1);
}

void QFramebuffer::updateFrame(float _f)
{
    setFrame((int)_f);
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

