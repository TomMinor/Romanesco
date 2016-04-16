#include "mainwindow.h"
#include <QApplication>
//#include <SDL.h>
//#include <SDL_haptic.h>

#include <SeExpression.h>

#include <QSurfaceFormat>
//#include <shaderwindow.h>

#include <mainwindow.h>

#include <iostream>
#include <vector>
#include <QDebug>


#include <OpenImageIO/imageio.h>


using namespace OpenImageIO;





class GrapherExpr:public SeExpression
{
public:
    //! Constructor that takes the expression to parse
    GrapherExpr(const std::string& expr)
        :SeExpression(expr)
    {}

    //! set the independent variable
    void setX(double x_input)
    {x.val=x_input;}

private:
    //! Simple variable that just returns its internal value
    struct SimpleVar:public SeExprScalarVarRef
    {
        double val; // independent variable
        void eval(const SeExprVarNode* /*node*/,SeVec3d& result)
        {result[0]=val;}
    };

    //! independent variable
    mutable SimpleVar x;

    //! resolve function that only supports one external variable 'x'
    SeExprVarRef* resolveVar(const std::string& name) const
    {
        if(name == "x") return &x;
        return 0;
    }
};





int main(int argc, char *argv[])
{
    std::string exprStr="\
                         $val=.5*PI*x;\
                         7*sin(val)/val \
                         ";
//    GrapherExpr expr(exprStr);

//    double xmin=-10,xmax=10,ymin=-10,ymax=10;
//    int w=60,h=30;
//    char *buffer=new char[w*h];
//    memset(buffer,(int)' ',w*h);

//    // draw x axis
//    int j_zero=(-ymin)/(ymax-ymin)*h;
//    if(j_zero>=0 && j_zero<h){
//        for(int i=0;i<w;i++){
//            buffer[i+j_zero*w]='-';
//        }
//    }
//    // draw y axis
//    int i_zero=(-xmin)/(xmax-xmin)*w;
//    if(i_zero>=0 && i_zero<w){
//        for(int j=0;j<h;j++){
//            buffer[i_zero+j*w]='|';
//        }
//    }

//    // evaluate the graph
//    const int samplesPerPixel=10;
//    const double one_over_samples_per_pixel=1./samplesPerPixel;
//    for(int i=0;i<w;i++)
//    {
//        for(int sample=0;sample<samplesPerPixel;sample++)
//        {
//            // transform from device to logical coordinatex
//            double dx=double(sample)*one_over_samples_per_pixel;
//            double x=double(dx+i)/double(w)*(xmax-xmin)+xmin;
//            // prep the expression engine for evaluation
//            expr.setX(x);
//            // evaluate and pull scalar value
//            SeVec3d val=expr.evaluate();
//            double y=val[0];
//            // transform from logical to device coordinate
//            int j=(y-ymin)/(ymax-ymin)*h;
//            // store to the buffer
//            if(j>=0 && j<h)
//                buffer[i+j*w]='#';
//        }
//    }

//    // draw the graph from the buffer
//    for(int j=h-1;j>=0;j--){
//        for(int i=0;i<w;i++){
//            std::cout<<buffer[i+j*w];
//        }
//        std::cout<<std::endl;
//    }



//  if (SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC) < 0 )
//  {
//    // Or die on error
//    qCritical("Unable to initialize SDL");
//  }

//  int numJoyPads = SDL_NumJoysticks();
//  if(numJoyPads ==0) {
//    qWarning( "No joypads found" );
//  } else {
//    qDebug( "Found %d joypads", numJoyPads );
//  }

  QApplication a(argc, argv);
//  MainWindow w;
//  w.show();

  QSurfaceFormat format;
  //format.setVersion(4, 3);
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setDepthBufferSize(24);
  format.setStencilBufferSize(8);
  QSurfaceFormat::setDefaultFormat(format);

  //haderWindow window;
  MainWindow window;
  //window.setFormat(format);
  window.resize(800, 600);
  window.show();

  //window.setAnimating(true);

  return a.exec();
}
