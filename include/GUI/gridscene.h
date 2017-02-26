#ifndef GRIDSCENE_H
#define GRIDSCENE_H

#include <QGraphicsScene>
#include <QDebug>
#include <QPainter>

class GridScene : public QGraphicsScene
{
public:
    GridScene(qreal x, qreal y, qreal w, qreal h)
        : QGraphicsScene(x, y, w, h)
    {
        QBrush brush(QColor(25, 25, 25));
        setBackgroundBrush(brush);
    }

protected:
    void drawBackground(QPainter *painter, const QRectF &rect)
    {
        QGraphicsScene::drawBackground(painter, rect);

        QPen pen = QPen(QColor(50, 50, 50), 1);
        painter->setPen( pen );

        const int gridSize = 25;

        qreal left = int(rect.left()) - (int(rect.left()) % gridSize);
        qreal top = int(rect.top()) - (int(rect.top()) % gridSize);

        QVarLengthArray<QLineF, 100> lines;

        for (qreal x = left; x < rect.right(); x += gridSize)
            lines.append(QLineF(x, rect.top(), x, rect.bottom()));
        for (qreal y = top; y < rect.bottom(); y += gridSize)
            lines.append(QLineF(rect.left(), y, rect.right(), y));

        painter->drawLines(lines.data(), lines.size());
    }
};

#endif // GRIDSCENE_H
