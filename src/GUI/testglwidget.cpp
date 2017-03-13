#include <QDir>

#include <QApplication>

#include "testglwidget.h"
#include "RenderMath.h"


TestGLWidget::TestGLWidget(QWidget *parent)
	: QOpenGLWidget(parent), m_optixScene(0), m_program(0), m_frame(0), m_time(0), m_previousWidth(0), m_previousHeight(0), m_debugLogger(nullptr)
{
//    m_desiredCamPos = m_camPos = QVector3D(1.09475, 0.0750364, -1.00239);
//    m_desiredCamRot = m_camRot = QVector3D(-0.301546, 0.399876, 0);

    m_desiredCamPos = m_camPos = QVector3D(1.5f,0.0,0);
    m_desiredCamRot = m_camRot = QVector3D(0,(0.5f * M_PI),0);

    m_updateCamera = true;

    m_overrideRes = false;

    m_timer = new QTimer(this);
    connect( m_timer, SIGNAL(timeout()), this, SLOT(updateScene()) );

}

TestGLWidget::~TestGLWidget()
{
    delete m_optixScene;
}

void TestGLWidget::updateScene()
{
//    m_optixScene->drawToBuffer();
}

void TestGLWidget::messageLogged(const QOpenGLDebugMessage &msg)
{
	QString error;

	// Format based on severity
	switch (msg.severity())
	{
	case QOpenGLDebugMessage::NotificationSeverity:
		error += "--";
		break;
	case QOpenGLDebugMessage::HighSeverity:
		error += "!!";
		break;
	case QOpenGLDebugMessage::MediumSeverity:
		error += "!~";
		break;
	case QOpenGLDebugMessage::LowSeverity:
		error += "~~";
		break;
	}

	error += " (";

	// Format based on source
#define CASE(c) case QOpenGLDebugMessage::c: error += #c; break
	switch (msg.source())
	{
		CASE(APISource);
		CASE(WindowSystemSource);
		CASE(ShaderCompilerSource);
		CASE(ThirdPartySource);
		CASE(ApplicationSource);
		CASE(OtherSource);
		CASE(InvalidSource);
	}
#undef CASE

	error += " : ";

	// Format based on type
#define CASE(c) case QOpenGLDebugMessage::c: error += #c; break
	switch (msg.type())
	{
		CASE(ErrorType);
		CASE(DeprecatedBehaviorType);
		CASE(UndefinedBehaviorType);
		CASE(PortabilityType);
		CASE(PerformanceType);
		CASE(OtherType);
		CASE(MarkerType);
		CASE(GroupPushType);
		CASE(GroupPopType);
	}
#undef CASE

	error += ")";
	qDebug() << qPrintable(error) << "\n" << qPrintable(msg.message()) << "\n";
}

void TestGLWidget::initializeGL()
{
    initializeOpenGLFunctions();

#define GL_DEBUG
#ifdef GL_DEBUG
	m_debugLogger = new QOpenGLDebugLogger(this);
	if (m_debugLogger->initialize())
	{
		qDebug() << "GL_DEBUG Debug Logger" << m_debugLogger << "\n";
		connect(m_debugLogger, SIGNAL(messageLogged(QOpenGLDebugMessage)), this, SLOT(messageLogged(QOpenGLDebugMessage)));
		m_debugLogger->startLogging();
	}
#endif // GL_DEBUG

	GLuint m_texId;
	glGenTextures(1, &m_texId);
	glBindTexture(GL_TEXTURE_2D, m_texId);

	// Change these to GL_LINEAR for super- or sub-sampling
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);

    glClearColor(1, 0, 0, 1);

    m_optixScene = nullptr;
    
	m_optixScene = new OptixScene(width(), height(), (QOpenGLFunctions_4_3_Core*)this );
    m_optixScene->setCameraType( OptixScene::CameraTypes::PINHOLE );

//    m_optixScene->setCurrentMaterial( "pathtrace_diffuse" );

    m_program = new QOpenGLShaderProgram(this);
//    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexSrc);
//    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentSrc);
    m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "D:/dev/RomanescoRenderer/shaders/viewport.vert");
    m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "D:/dev/RomanescoRenderer/shaders/viewport.frag");

    if(!m_program->link())
    {
        qDebug() << "Link error in shader program\n";
        qDebug() << m_program->log();
		std::runtime_error("Link error in shader program");
    }

    
//    m_resXUniform = m_program->uniformLocation("resx");
//    m_resYUniform = m_program->uniformLocation("resy");
//    m_aspectUniform = m_program->uniformLocation("aspect");
//    m_timeUniform = m_program->uniformLocation("time");
//    m_posUniform = m_program->uniformLocation("pos");
//    m_normalMatrix = m_program->uniformLocation("normalMatrix");

	static GLfloat vertices[] = {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f, 1.0f,
		-1.0f, -1.0f,
		-1.0f, 1.0f,
		1.0f, 1.0f
	};
	static GLfloat uv[] = {
		0, 0,
		1, 0,
		1, 1,
		0, 0,
		0, 1,
		1, 1
	};

	// Setup VBO ready for drawing
	{
		glGenVertexArrays(1, &m_vao);
		glBindVertexArray(m_vao);

		glGenBuffers(1, &m_vbo);
		
		// Upload vertex data to GPU
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		m_program->bind();

		m_vtxPosAttr = m_program->attributeLocation("vtxPos");
		if (m_vtxPosAttr == -1)
		{
			qWarning() << "Couldn't find 'vtxPos' attribute";
			//throw std::runtime_error("Couldn't find 'vtxPos' attribute");
		}

		m_vtxUVAttr = m_program->attributeLocation("vtxUV");
		if (m_vtxUVAttr == -1)
		{
			qWarning() << "Couldn't find 'vtxUV' attribute";
			//throw std::runtime_error("Couldn't find 'vtxPos' attribute");
		}

		glVertexAttribPointer(m_vtxPosAttr, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(m_vtxPosAttr);

		glVertexAttribPointer(m_vtxUVAttr, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(m_vtxUVAttr);

		m_program->release();
	}

    emit initializedGL();
}

void TestGLWidget::resizeGL(int _w, int _h)
{
    if(m_overrideRes && m_previousHeight == m_overrideHeight && m_previousWidth == m_overrideWidth )
    {
        return;
    }
//        m_projection.setToIdentity();
//        m_projection.perspective(60.0f, w / float(h), 0.01f, 1000.0f);

//    qDebug() << m_previousHeight << height() << ", " << m_previousWidth << width() << ", " << m_updateCamera;
    int w = width();
    int h = height();

    if( m_previousHeight != h || m_previousWidth != w )
    {
        if(m_overrideRes)
        {
            w = m_overrideWidth;
            h = m_overrideHeight;
        }

        m_previousHeight = h;
        m_previousWidth = w;

        if(m_optixScene)
        {
            m_updateCamera = true;
            m_optixScene->updateBufferSize( w, h );
        }
    } else {
        m_updateCamera = false;
    }

    m_timer->start(0);
}

#define EPSILON 0.01

bool AreSame(double a, double b)
{
    return std::fabs(a - b) < EPSILON;
}

bool AreSame(QVector3D a, QVector3D b)
{
    return (    fabs(a.x() - b.x()) < EPSILON &&
                fabs(a.y() - b.y()) < EPSILON &&
                fabs(a.z() - b.z()) < EPSILON );
}

void TestGLWidget::updateCamera()
{
	if (!m_optixScene)
		return;

    m_optixScene->setCamera(  optix::make_float3( m_camPos.x(), m_camPos.y(), m_camPos.z() ),
							  optix::make_float3(0.0f, 0.3f, 0.0f),
                              m_fov,
                              m_overrideRes ? m_overrideWidth : width(),
                              m_overrideRes ? m_overrideHeight : height()
                              );
}

void TestGLWidget::paintGL()
{
    // Quickfix to show window before painting
    static bool isInit = false;

    if(!isInit)
    {
        isInit = true;
        return;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_camRot = m_desiredCamRot;
    m_camPos = m_desiredCamPos;

	optix::Matrix4x4 rotX = rotX.rotate(m_camRot.x(), optix::make_float3(1, 0, 0));
	optix::Matrix4x4 rotY = rotY.rotate(m_camRot.y(), optix::make_float3(0, 1, 0));
	optix::Matrix4x4 rotZ = rotZ.rotate(m_camRot.z(), optix::make_float3(0, 0, 1));
    optix::Matrix4x4 normalmatrix = rotX * rotY * rotZ;

//    m_camPos.setX( FInterpTo( m_camPos.x(), m_desiredCamPos.x(), m_frame, 0.0001) );
//    m_camPos.setY( FInterpTo( m_camPos.y(), m_desiredCamPos.y(), m_frame, 0.0001) );
//    m_camPos.setZ( FInterpTo( m_camPos.z(), m_desiredCamPos.z(), m_frame, 0.0001) );

//    m_camRot.setX( FInterpTo( m_camRot.x(), m_desiredCamRot.x(), m_frame, 0.00025) );
//    m_camRot.setY( FInterpTo( m_camRot.y(), m_desiredCamRot.y(), m_frame, 0.00025) );
//    m_camRot.setZ( FInterpTo( m_camRot.z(), m_desiredCamRot.z(), m_frame, 0.00025) );


    if(m_optixScene)
    {

        if(m_updateCamera)
        {
            qDebug() << m_camPos;
            qDebug() << m_camRot;

            m_optixScene->setVar("normalmatrix", normalmatrix);
            updateCamera();
            m_updateCamera = false;
        }
    }

//    if(AreSame(m_camPos, m_desiredCamPos) && AreSame(m_camRot, m_desiredCamRot))
//    {
//        m_updateCamera = false;
//    } else {
//        m_updateCamera = true;
//    }


    const qreal retinaScale = devicePixelRatio();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);
    glClear(GL_COLOR_BUFFER_BIT);
	
    //glEnable(GL_TEXTURE_2D);

	m_program->bind();
	glBindVertexArray(m_vao);
	
    if(m_optixScene)
    {
        m_optixScene->drawToBuffer();
    }
	
	glDrawArrays(GL_TRIANGLES, 0, 6);

	m_program->release();
    //glDisable(GL_TEXTURE_2D);



//    ++m_frame;
}

void TestGLWidget::mouseMoveEvent(QMouseEvent* _event)
{

}

void TestGLWidget::mousePressEvent(QMouseEvent* _event)
{

}

void TestGLWidget::mouseReleaseEvent(QMouseEvent* _event)
{

}

void TestGLWidget::enterEvent(QEvent* _event)
{
    this->grabKeyboard();
}

void TestGLWidget::leaveEvent(QEvent* _event)
{
    this->releaseKeyboard();
}

void TestGLWidget::keyPressEvent(QKeyEvent *_event)
{
    const float offset = 0.025f;
    const float rotateOffset = 0.10f;

	if (!m_optixScene)
		return;

    switch ( _event->key() )
    {
        case Qt::Key_1:
        {
            m_optixScene->setOutputBuffer("output_buffer");
            break;
        }
        case Qt::Key_2:
        {
            m_optixScene->setOutputBuffer("output_buffer_nrm");
            break;
        }
        case Qt::Key_3:
        {
            m_optixScene->setOutputBuffer("output_buffer_world");
            break;
        }
        case Qt::Key_4:
        {
            m_optixScene->setOutputBuffer("output_buffer_depth");
            break;
        }
        case Qt::Key_5:
        {
            m_optixScene->setOutputBuffer("output_buffer_trap");
            break;
        }
        case Qt::Key_6:
        {
            m_optixScene->setOutputBuffer("output_buffer_iteration");
            break;
        }
        case Qt::Key_7:
        {
            m_optixScene->setOutputBuffer("output_buffer_diffuse");
            break;
        }

        case Qt::Key_A:
            {
                float radius = offset;
                float pitch = m_camRot.x();
                float yaw = m_camRot.y() + ( M_PI/2.0f );

                float pitchRad = pitch;//qDegreesToRadians( pitch );
                float yawRad = yaw;//qDegreesToRadians( yaw );

                //These equations are from the wikipedia page, linked above
                float xMove = radius * -sinf( yawRad ) * cosf( pitchRad );
                float yMove = radius * sinf( pitchRad );
                float zMove = radius * cosf( yawRad ) * cosf( pitchRad );

                m_desiredCamPos.setX( m_desiredCamPos.x() - xMove );
                m_desiredCamPos.setY( m_desiredCamPos.y() - yMove );
                m_desiredCamPos.setZ( m_desiredCamPos.z() - zMove );
                m_updateCamera = true;
            }

            break;

        case Qt::Key_D:
            {
                float radius = offset;
                float pitch = m_camRot.x();
                float yaw = m_camRot.y() + ( M_PI/2.0f );

                float pitchRad = pitch;//qDegreesToRadians( pitch );
                float yawRad = yaw;//qDegreesToRadians( yaw );

                //These equations are from the wikipedia page, linked above
                float xMove = radius * -sinf( yawRad ) * cosf( pitchRad );
                float yMove = radius * sinf( pitchRad );
                float zMove = radius * cosf( yawRad ) * cosf( pitchRad );

                m_desiredCamPos.setX( m_desiredCamPos.x() + xMove );
                m_desiredCamPos.setY( m_desiredCamPos.y() + yMove );
                m_desiredCamPos.setZ( m_desiredCamPos.z() + zMove );
                m_updateCamera = true;
            }

            break;

        case Qt::Key_W:
            {
                float radius = offset;
                float pitch = m_camRot.x();
                float yaw = m_camRot.y();

                float pitchRad = pitch;//qDegreesToRadians( pitch );
                float yawRad = yaw;//qDegreesToRadians( yaw );

                //These equations are from the wikipedia page, linked above
                float xMove = radius * -sinf( yawRad ) * cosf( pitchRad );
                float yMove = radius * sinf( pitchRad );
                float zMove = radius * cosf( yawRad ) * cosf( pitchRad );

                m_desiredCamPos.setX( m_desiredCamPos.x() + xMove );
                m_desiredCamPos.setY( m_desiredCamPos.y() + yMove );
                m_desiredCamPos.setZ( m_desiredCamPos.z() + zMove );
                m_updateCamera = true;
            }

            break;

        case Qt::Key_S:
            {
                float radius = offset;
                float pitch = m_camRot.x();
                float yaw = m_camRot.y();

                float pitchRad = pitch;// qDegreesToRadians( pitch );
                float yawRad = yaw;//qDegreesToRadians( yaw );

                //These equations are from the wikipedia page, linked above
                float xMove = radius * -sinf( yawRad ) * cosf( pitchRad );
                float yMove = radius * sinf( pitchRad );
                float zMove = radius * cosf( yawRad ) * cosf( pitchRad );

                m_desiredCamPos.setX( m_desiredCamPos.x() - xMove );
                m_desiredCamPos.setY( m_desiredCamPos.y() - yMove );
                m_desiredCamPos.setZ( m_desiredCamPos.z() - zMove );
                m_updateCamera = true;
            }

            break;

        case Qt::Key_Up:
            {
                m_desiredCamRot.setX( m_desiredCamRot.x() + rotateOffset );
                m_updateCamera = true;
            }
            break;

        case Qt::Key_Down:
            {
                m_desiredCamRot.setX( m_desiredCamRot.x() - rotateOffset );
                m_updateCamera = true;
            }
            break;

        case Qt::Key_Left:
            {
                m_desiredCamRot.setY( m_desiredCamRot.y() - rotateOffset );
                m_updateCamera = true;
            }
            break;

        case Qt::Key_Right:
            {
                 m_desiredCamRot.setY( m_desiredCamRot.y() + rotateOffset );
                 m_updateCamera = true;
            }
            break;

        case Qt::Key_P:
            {
                m_optixScene->saveBuffersToDisk("./text.exr");
            }
            break;
        case Qt::Key_U:
            {
                qDebug() << "Updating geo";
                system("nvcc -m64 -gencode arch=compute_20,code=sm_20 --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -ptx -I$OPTIX_PATH/SDK/sutil -I$OPTIX_PATH/SDK -I$OPTIX_PATH/include kernel/menger.cu -o ptx/menger.cu.ptx");
                m_optixScene->createCameras();
                m_optixScene->createWorld();
                m_updateCamera = true;
            }
            break;
        default:
            _event->ignore();
            break;
    }
}
