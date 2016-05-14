#include "include/GUI/qframebuffer.h"

#include <QLayout>
#include <QMenuBar>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>

QFramebuffer::QFramebuffer(QWidget *parent) : QMainWindow(parent)
{
    QWidget* widget = new QWidget;
    QVBoxLayout* layout = new QVBoxLayout;
    //    QVBoxLayout* layout = new QVBoxLayout;

    m_menu = new QMenuBar(this);
    this->setMenuBar( m_menu );

    m_timeline = new QAnimatedTimeline;

    QImage image("/home/tom/Pictures/fractal.jpg");

    m_view = new QGraphicsView;

    m_scene = new QGraphicsScene;
    m_scene->addPixmap( QPixmap::fromImage(image) );

    QRectF tmp(0, 0, image.width(), image.height());
    m_scene->setSceneRect(tmp);

    m_view->setScene(m_scene);

    layout->addWidget(m_view);
    layout->addWidget(m_timeline);

    widget->setLayout(layout);

    this->setCentralWidget(widget);

//    this->setCentralWidget(m_view);
//    this->setLayout( layout );
}

