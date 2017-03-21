#include <QDebug>

#include "ImageWriter.h"
#include <cassert>

bool ImageWriter::progressCallback(void* _data, float _progress)
{
    std::string name = "unknown";
    if(_data)
    {
        ImageWriter* writer = nullptr;
        if( (writer = reinterpret_cast<ImageWriter*>(_data)) )
        {
            name = writer->getFileName();
        }
    }

    qDebug("Writing image (%s) : %.2f%% complete", name.c_str(), (_progress * 100.0));

    return false;
}



ImageWriter::ImageWriter(std::string _filename, unsigned int _width, unsigned int _height)
	: m_width(_width), m_height(_height), m_header(_width, _height), m_fileName(_filename)
{
	;
}

ImageWriter::~ImageWriter()
{
//    if(m_outFile)
//    {
//        m_outFile->close();
////        OpenImageIO::ImageOutput::destroy(m_outFile);
//
//        m_outFile = nullptr;
//    }
//
//    if(m_spec)
//    {
//        delete m_spec;
//    }
}


std::string ImageWriter::layerChannelString(std::string _layerName, std::string _channel) const
{
	return (_layerName.length() == 0) ? _channel : _layerName + "." + _channel;
}


bool ImageWriter::write(std::vector<Romanesco::Channel> _channels)
{
	Imf::ChannelList& channels = m_header.channels();

	Romanesco::Channel& img = _channels[0];
	//// The interop input buffers are all float based (float3, float4), so we divide by 4 bytes
	unsigned int elements = img.m_elementSize / sizeof(float);

	addChannel(img);

	Imf::OutputFile file(m_fileName.c_str(), m_header);
	file.setFrameBuffer(m_framebuffer);
	file.writePixels(img.m_height);

//    m_outFile = nullptr;
//    m_outFile = OpenImageIO::ImageOutput::create(m_fileName);
//
//    if(!m_outFile)
//    {
//        return false;
//    }
//
//    OpenImageIO::ProgressCallback callback = ImageWriter::progressCallback;
//    m_outFile->open(m_fileName, *m_spec);
//
//    bool success = false;
//
//    // Double check that this format accepts per-channel formats
//    if (!m_outFile->supports("channelformats")) {
//        qDebug("Warning: Filetype %s doesn't support multichannel data", m_outFile->format_name());
//
////        OpenImageIO::ImageOutput::destroy(m_outFile);
//        m_outFile->close();
//        throw std::runtime_error("Can't write per channel data (passes) in ImageWriter");
//    }
//
//    qDebug() << sizeof( ImageWriter::Pixel );
//
//    success = m_outFile->write_image (OpenImageIO::TypeDesc::UNKNOWN,
//                            _pixels.data(),
//                            sizeof( ImageWriter::Pixel ),
//                            OpenImageIO::AutoStride,
//                            OpenImageIO::AutoStride,
//                            callback,
//                            this // This class is the callback payload
//                            );
//
//    m_outFile->close();
//
//
//    return success;

	return false;
}


void ImageWriter::addChannel(Romanesco::Channel& _img)
{
	Imf::ChannelList& channels = m_header.channels();
	// The interop input buffers are all float based (float3, float4), so we divide by 4 bytes
	unsigned int elements = _img.m_elementSize / sizeof(float);
	
	std::string name = _img.m_name;
	char* rPtr = (char*)&(_img.m_pixels[0].r);
	char* gPtr = (char*)&(_img.m_pixels[0].g);
	char* bPtr = (char*)&(_img.m_pixels[0].b);
	char* aPtr = (char*)&(_img.m_pixels[0].a);

	//qDebug() << name.c_str() << elements;

	switch (elements)
	{
	case 1:	addChannel(name, rPtr); break;
	case 3:	addChannelRGB(name, rPtr, gPtr, bPtr); break;
	case 4:	addChannelRGBA(name, rPtr, gPtr, bPtr, aPtr); break;
	default:
	{
		qDebug() << "Can't add channel of size " + elements;
		assert(false && "Can't add channel of size " + elements);
	}
	}
}

void ImageWriter::addChannel(std::string _name, char* _pixels)
{
	qDebug() << "Adding data channel " << _name.c_str();

	assert(_name.length() > 0);

	const char* name = _name.c_str();

	Imf::ChannelList& channels = m_header.channels();
	// The interop input buffers are all float based (float3, float4), so we divide by 4 bytes
	unsigned int elements = 1;

	channels.insert(name, Imf::Channel(Imf::HALF));
	m_framebuffer.insert(name, Imf::Slice(Imf::HALF, _pixels, sizeof(half) * elements, sizeof(half) * elements * m_width));
}

void ImageWriter::addChannelRGB(std::string _name, char* _pixelsR, char* _pixelsB, char* _pixelsG)
{
	const char* name = _name.c_str();
	qDebug() << "Adding RGB channel " << name;

	Imf::ChannelList& channels = m_header.channels();

	unsigned int elements = 3;// _img.m_elementSize / sizeof(float);

	channels.insert(layerChannelString(name, "R").c_str(), Imf::Channel(Imf::HALF));
	channels.insert(layerChannelString(name, "G").c_str(), Imf::Channel(Imf::HALF));
	channels.insert(layerChannelString(name, "B").c_str(), Imf::Channel(Imf::HALF));

	m_framebuffer.insert(layerChannelString(name, "R").c_str(), Imf::Slice(Imf::HALF, _pixelsR, sizeof(half) * elements, sizeof(half) * elements * m_width));
	m_framebuffer.insert(layerChannelString(name, "G").c_str(), Imf::Slice(Imf::HALF, _pixelsG, sizeof(half) * elements, sizeof(half) * elements * m_width));
	m_framebuffer.insert(layerChannelString(name, "B").c_str(), Imf::Slice(Imf::HALF, _pixelsB, sizeof(half) * elements, sizeof(half) * elements * m_width));
}

void ImageWriter::addChannelRGBA(std::string _name, char* _pixelsR, char* _pixelsB, char* _pixelsG, char* _pixelsA)
{
	const char* name = _name.c_str();
	qDebug() << "Adding RGBA channel " << name;

	Imf::ChannelList& channels = m_header.channels();

	unsigned int elements = 4;// / sizeof(float);

	channels.insert(layerChannelString(name, "R").c_str(), Imf::Channel(Imf::HALF));
	channels.insert(layerChannelString(name, "G").c_str(), Imf::Channel(Imf::HALF));
	channels.insert(layerChannelString(name, "B").c_str(), Imf::Channel(Imf::HALF));
	channels.insert(layerChannelString(name, "A").c_str(), Imf::Channel(Imf::HALF));

	m_framebuffer.insert(layerChannelString(name, "R").c_str(), Imf::Slice(Imf::HALF, _pixelsR, sizeof(half) * elements, sizeof(half) * elements * m_width));
	m_framebuffer.insert(layerChannelString(name, "G").c_str(), Imf::Slice(Imf::HALF, _pixelsG, sizeof(half) * elements, sizeof(half) * elements * m_width));
	m_framebuffer.insert(layerChannelString(name, "B").c_str(), Imf::Slice(Imf::HALF, _pixelsB, sizeof(half) * elements, sizeof(half) * elements * m_width));
	m_framebuffer.insert(layerChannelString(name, "A").c_str(), Imf::Slice(Imf::HALF, _pixelsA, sizeof(half) * elements, sizeof(half) * elements * m_width));
}


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