

#include <QApplication>
////#include <SDL.h>
////#include <SDL_haptic.h>
//
//#include <SeExpression.h>

#include <QSurfaceFormat>
////#include <shaderwindow.h>

#include <OpenEXR/ImfRgba.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/half.h>

//#include <OpenImageIO\imageio.h>

#include <cuda_runtime.h>
#include <cuda.h>


#ifdef _WIN32
#define NOMINMAX
#endif
#include <optix.h>
#include <optixu/optixu.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixpp_namespace.h>



#include <iostream>
#include <vector>
#include <QDebug>

#include "mainwindow.h"
#include "qframebuffer.h"

#include <future>
#include <iostream>

//#include <boost/algorithm/string/join.hpp>

//void test()
//{
//	/*half a;
//
//	auto arse = new OpenImageIO::ImageSpec(OpenImageIO::TypeDesc::UNKNOWN);
//*/
//
//	optix::Buffer a;
//
//	cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyDeviceToHost);
//
//	std::vector<std::string> list;
//	list.push_back("Hello");
//	list.push_back("World!");
//
//	std::string joined = boost::algorithm::join(list, ", ");
//	std::cout << joined << std::endl;
//}
//
//struct Image
//{
//public:
//	Image(float* _pixels, unsigned int _width, unsigned int _height, std::string _name = "")
//		: m_width(_width), m_height(_height), m_name(_name)
//	{
//		m_pixels = new Imf::Rgba[m_width * m_height];
//		std::fill(m_pixels, m_pixels + (m_width * m_height), Imf::Rgba(1.f, 1.f, 1.f, 1.f));
//
//		for (int i = 0; i < 4 * m_width * m_height; i += 4)
//		{
//			//unsigned int idx = i + (j * m_width);
//
//			float R = _pixels[i];
//			float G = _pixels[i + 1];
//			float B = _pixels[i + 2];
//			float A = _pixels[i + 3];
//
//			//setPixel(i, j, Imf::Rgba(R, G, B, A) );
//			m_pixels[i / 4] = Imf::Rgba(R, G, B, A);
//		}
//	}
//
//	void setPixel(int x, int y, Imf::Rgba _val)
//	{
//		m_pixels[x + (y * m_width)] = _val;
//	}
//
//	~Image()
//	{
//		//delete m_pixels;
//	}
//
//	//private:
//	Imf::Rgba* m_pixels;
//	unsigned int m_width, m_height;
//	std::string m_name;
//};
//
//std::string layerChannelString( std::string _layerName, std::string _channel )
//{
//    return (_layerName.size() == 0) ? _channel : _layerName + "." + _channel;
//}
//
//void writeRGBA2(std::string fileName, std::vector<Image> _layers)
//{
//	Imf::Header header(_layers[0].m_width, _layers[0].m_height);
//
//	Imf::ChannelList& channels = header.channels();
//	Imf::FrameBuffer framebuffer;
//
//	for (unsigned int i = 0; i < _layers.size(); i++)
//	{
//		Image& _image = _layers[i];
//
//		std::string name_r = layerChannelString(_image.m_name, "R");
//		std::string name_g = layerChannelString(_image.m_name, "G");
//		std::string name_b = layerChannelString(_image.m_name, "B");
//		std::string name_a = layerChannelString(_image.m_name, "A");
//
//		channels.insert(name_r, Imf::Channel(Imf::HALF));
//		channels.insert(name_g, Imf::Channel(Imf::HALF));
//		channels.insert(name_b, Imf::Channel(Imf::HALF));
//		channels.insert(name_a, Imf::Channel(Imf::HALF));
//
//		char* channel_rPtr = (char*)&(_image.m_pixels[0].r);
//		char* channel_gPtr = (char*)&(_image.m_pixels[0].g);
//		char* channel_bPtr = (char*)&(_image.m_pixels[0].b);
//		char* channel_aPtr = (char*)&(_image.m_pixels[0].a);
//
//		unsigned int xstride = sizeof(half) * 4;
//		unsigned int ystride = sizeof(half) * 4 * _image.m_width;
//
//		framebuffer.insert(name_r, Imf::Slice(Imf::HALF, channel_rPtr, xstride, ystride));
//		framebuffer.insert(name_g, Imf::Slice(Imf::HALF, channel_gPtr, xstride, ystride));
//		framebuffer.insert(name_b, Imf::Slice(Imf::HALF, channel_bPtr, xstride, ystride));
//		framebuffer.insert(name_a, Imf::Slice(Imf::HALF, channel_aPtr, xstride, ystride));
//	}
//
//	Imf::OutputFile file(fileName.c_str(), header);
//	file.setFrameBuffer(framebuffer);
//	file.writePixels(_layers[0].m_height);
//	
//}


int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	QCoreApplication::setApplicationName("Romanesco Renderer");
	QCoreApplication::setApplicationVersion("2.0");

	QSurfaceFormat format;
	format.setVersion(4, 3);
	format.setProfile(QSurfaceFormat::CoreProfile);
	format.setOption(QSurfaceFormat::DebugContext);
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	QSurfaceFormat::setDefaultFormat(format);

	MainWindow window;
	//window.setFormat(format);

	// Load and set stylesheet
	QFile file("styles/romanesco.qss");
	file.open(QFile::ReadOnly);
	QString stylesheet = QLatin1String(file.readAll());
	window.setGlobalStyleSheet(stylesheet);

	window.setWindowTitle("Romanesco");
	window.resize(1280, 800);
	window.show();

	//  window.setAnimating(true);

	app.exec();
}
