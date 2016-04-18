#include <QStylePainter>
#include <QStyleOptionSlider>

#include "qtimeslider.h"

void QTimeSlider::paintEvent(QPaintEvent *ev)
{
    QStylePainter p(this);
    QStyleOptionSlider opt;
    initStyleOption(&opt);

    QFont font = p.font();
    font.setPointSize(12);
    p.setFont(font);

    QRect handle = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, this);

    // draw tick marks
    // do this manually because they are very badly behaved with style sheets
    int interval = tickInterval();
    if (interval == 0)
    {
        interval = pageStep();
    }

    if (tickPosition() != NoTicks)
    {
        for (int i = minimum(); i <= maximum(); i += interval)
        {
            int x = round((double)((double)((double)(i - this->minimum()) / (double)(this->maximum() - this->minimum())) * (double)(this->width() - handle.width()) + (double)(handle.width() / 2.0))) - 1;
            int h = 4;

            p.setPen(QColor("#000000"));
            p.drawText(QPointF(x - 6, this->rect().bottom() - 0), QString::number(i) );

            p.setPen(QColor("#a5a294"));
            if (tickPosition() == TicksBothSides || tickPosition() == TicksAbove)
            {
                int y = this->rect().top();
                p.drawLine(x, y, x, y + h);
            }
            if (tickPosition() == TicksBothSides || tickPosition() == TicksBelow)
            {
                int y = this->rect().bottom();
                p.drawLine(x, y, x, y - h);
            }
        }
    }

    // draw the slider (this is basically copy/pasted from QSlider::paintEvent)
    opt.subControls = QStyle::SC_SliderGroove;
    p.drawComplexControl(QStyle::CC_Slider, opt);

    // draw the slider handle
    opt.subControls = QStyle::SC_SliderHandle;
    p.drawComplexControl(QStyle::CC_Slider, opt);
}
