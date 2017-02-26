/* Copyright (c) 2012, STANISLAW ADASZEWSKI
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of STANISLAW ADASZEWSKI nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL STANISLAW ADASZEWSKI BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

#pragma once

#include "QNENodes/qneblock.h"
#include "QNENodes/qneport.h"

#include <QPen>
#include <QGraphicsScene>
#include <QFontMetrics>
#include <QPainter>


QNEBlock::QNEBlock(QGraphicsScene *_scene, QGraphicsItem *parent) : QGraphicsPathItem(parent)
{
	QPainterPath p;
    p.addRoundedRect(-50, -15,
                     60, 100,
                     5, 5);
	setPath(p);
	setPen(QPen(Qt::darkGreen));
	setBrush(Qt::green);
	setFlag(QGraphicsItem::ItemIsMovable);
	setFlag(QGraphicsItem::ItemIsSelectable);
    setOpacity(0.9f);
	horzMargin = 20;
	vertMargin = 5;
	width = horzMargin;
	height = vertMargin;

    _scene->addItem(this);
}

#include <QDebug>

QNEPort* QNEBlock::addPort(const QString &name, bool isOutput, int flags, int ptr)
{
	QNEPort *port = new QNEPort(this);
	port->setName(name);
    port->setIsOutput(isOutput);
	port->setNEBlock(this);
    port->setPortFlags(flags);
	port->setPtr(ptr);

	QFontMetrics fm(scene()->font());
	int w = fm.width(name);
	int h = fm.height();
    //port->setPos(0, height + h/2);
	if (w > width - horzMargin)
		width = w + horzMargin;
	height += h;

    QPainterPath p;
    p.addRoundedRect(-width/2, -height/2, width, height, 2, 2);
    setPath(p);

    int inputCtr = 0;
    int outputCtr = 0;
	int y = -height / 2 + vertMargin + port->radius();
    foreach(QGraphicsItem *port_, childItems())
    {
		if (port_->type() != QNEPort::Type)
			continue;

		QNEPort *port = (QNEPort*) port_;
        if(!port->isIO())
        {
            // Name ports etc
            port->setPos(-width/2 - port->radius(), y);
            y += h;
            continue;
        }
        else
        {
            if(port->isOutput())
            {
                outputCtr++;
            } else {
                inputCtr++;
            }
        }
	}

    if(outputCtr == 0 && inputCtr == 0)
    {
        return port;
    }

    const float portSpacing = 0.0f;
    const float portOffset = (h + (portSpacing * port->radius()));
    float yIn = portOffset;//y + portOffset;
    float yOut = portOffset;//y + portOffset;

    if(inputCtr < outputCtr)
    {
        float offsetRatio = (inputCtr > 0) ? (float)outputCtr / (float)inputCtr : 0.0f;

//        yIn += (offsetRatio * portOffset);
//        qDebug() << "In " << yIn ;
    }
    else if(inputCtr != outputCtr)
    {
        float offsetRatio = (outputCtr > 0) ? (float)inputCtr / (float)outputCtr : 0.0f;

//        yOut += (offsetRatio * portOffset);
//        qDebug() << "Out " << yOut ;
    }

    foreach(QGraphicsItem *port_, childItems()) {
        if (port_->type() != QNEPort::Type)
            continue;

        QNEPort *port = (QNEPort*) port_;
        if(port->isIO())
        {
            if (port->isOutput())
            {
                port->setPos(width/2 + port->radius(), yOut);
                yOut += portOffset;
            }
            else
            {
                port->setPos(-width/2 - port->radius(), yIn);
                yIn += portOffset;
            }
        }
    }

	return port;
}

QString QNEBlock::displayName()
{
    foreach(QGraphicsItem *port_, childItems()) {
        QNEPort *port = qgraphicsitem_cast<QNEPort*>(port_);
        if(!port)
        {
            continue;
        }
        if(port->isDisplayName())
        {
            return port->portName();
        }
    }

    return QString();
}

QString QNEBlock::typeName()
{
    foreach(QGraphicsItem *port_, childItems()) {
        QNEPort *port = qgraphicsitem_cast<QNEPort*>(port_);
        if(!port)
        {
            continue;
        }
        if(port->isTypeName())
        {
            return port->portName();
        }
    }

    return QString();
}

void QNEBlock::addInputPort(const QString &name)
{
	addPort(name, false);
}

void QNEBlock::addOutputPort(const QString &name)
{
	addPort(name, true);
}

void QNEBlock::addInputPorts(const QStringList &names)
{
	foreach(QString n, names)
		addInputPort(n);
}

void QNEBlock::addOutputPorts(const QStringList &names)
{
	foreach(QString n, names)
		addOutputPort(n);
}

void QNEBlock::save(QDataStream &ds)
{
	ds << pos();

	int count(0);

    foreach(QGraphicsItem *port_, childItems())
	{
		if (port_->type() != QNEPort::Type)
			continue;

		count++;
	}

	ds << count;

    foreach(QGraphicsItem *port_, childItems())
	{
		if (port_->type() != QNEPort::Type)
			continue;

		QNEPort *port = (QNEPort*) port_;
		ds << (quint64) port;
		ds << port->portName();
		ds << port->isOutput();
		ds << port->portFlags();
	}
}

void QNEBlock::load(QDataStream &ds, QMap<quint64, QNEPort*> &portMap)
{
	QPointF p;
	ds >> p;
	setPos(p);
	int count;
	ds >> count;
	for (int i = 0; i < count; i++)
	{
		QString name;
		bool output;
		int flags;
		quint64 ptr;

		ds >> ptr;
		ds >> name;
		ds >> output;
		ds >> flags;
		portMap[ptr] = addPort(name, output, flags, ptr);
	}
}

#include <QStyleOptionGraphicsItem>

void QNEBlock::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)

    if (isSelected()) {
        painter->setPen(QPen( QColor(255 * 0.5, 153 * 0.5, 0) ));
        painter->setBrush( QColor(255, 153, 0) );
    } else {
        painter->setPen(QPen(QPen( QColor( 50, 50, 50 ) ) ) );
        painter->setBrush( QColor( 135, 135, 135 ) );
    }

    painter->drawPath( path() );
}

QNEBlock* QNEBlock::clone()
{
    QNEBlock *b = new QNEBlock(0);
    this->scene()->addItem(b);

	foreach(QGraphicsItem *port_, childItems())
	{
		if (port_->type() == QNEPort::Type)
		{
			QNEPort *port = (QNEPort*) port_;
			b->addPort(port->portName(), port->isOutput(), port->portFlags(), port->ptr());
		}
	}

	return b;
}

QVector<QNEPort*> QNEBlock::ports()
{
	QVector<QNEPort*> res;
	foreach(QGraphicsItem *port_, childItems())
	{
		if (port_->type() == QNEPort::Type)
			res.append((QNEPort*) port_);
	}
	return res;
}

QVector<QNEPort*> QNEBlock::inputPorts()
{
    QVector<QNEPort*> res;
    foreach(QGraphicsItem *port_, childItems())
    {
        QNEPort* port = qgraphicsitem_cast<QNEPort*>(port_);
        if(port)
        {
            GraphicsItemFlags flags = port->flags();

            // Don't add name ports etc
            if( port->isIO() )
            {
                if(!port->isOutput())
                {
                    res.append(port);
                }
            }
        }
    }

    return res;
}

#include <QDebug>

QVector<QNEPort*> QNEBlock::outputPorts()
{
    QVector<QNEPort*> res;
    foreach(QGraphicsItem *port_, childItems())
    {
        QNEPort* port = qgraphicsitem_cast<QNEPort*>(port_);
        if(port)
        {
            GraphicsItemFlags flags = port->flags();

            // Don't add name ports etc
            if( !((flags & QNEPort::NamePort)
               || (flags & QNEPort::TypePort)) )
            {
                if(port->isOutput())
                {
                    res.append(port);
                }
            }
        }
    }

    return res;
}

QVariant QNEBlock::itemChange(GraphicsItemChange change, const QVariant &value)
{

    Q_UNUSED(change);

	return value;
}

