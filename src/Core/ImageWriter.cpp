#include <QDebug>

#include "ImageWriter.h"


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
    : /*m_outFile(nullptr), m_spec(nullptr), */m_fileName(_filename)
{
//    m_spec = new OpenImageIO::ImageSpec(OpenImageIO::TypeDesc::UNKNOWN);
//
//    m_spec->channelnames.clear();
//    m_spec->attribute("compression", "zip");
//
//
//    // Channel setup
//    addChannelRGBA(OpenImageIO::TypeDesc::HALF);        // RGBA Channels
//    ///@todo Somehow get this to higher quality precision
//    addChannel(OpenImageIO::TypeDesc::HALF, "Z");     // Depth Channel
//
//    addChannel(OpenImageIO::TypeDesc::HALF, "orbit.R");     // Depth Channel
//    addChannel(OpenImageIO::TypeDesc::HALF, "orbit.G");     // Depth Channel
//    addChannel(OpenImageIO::TypeDesc::HALF, "orbit.B");     // Depth Channel
//    addChannel(OpenImageIO::TypeDesc::HALF, "iteration.R");     // Normalized Iteration Channel
//    addChannelRGB(OpenImageIO::TypeDesc::HALF, "N");    // Normal Channels
//    addChannelRGB(OpenImageIO::TypeDesc::HALF, "P");    // World Position Channels
//
//    addChannelRGB(OpenImageIO::TypeDesc::HALF, "diffuse");
//
//
//    qDebug() << sizeof(ImageWriter::Pixel) << "," << m_spec->channelnames.size();
//
//    m_spec->nchannels = m_spec->channelnames.size();
//    m_spec->width = _width;
//    m_spec->height = _height;
//
//    // Tell OIIO that these specific channels will store alpha/depth (helps for some formats)
//    m_spec->alpha_channel = 3;
//    m_spec->z_channel = 4;
//
//    // Handy for debugging later?
//    m_spec->attribute("Romanesco Version", 0.1f);
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

/// @todo Move this into Channel class
std::string layerChannelString(std::string _layerName, std::string _channel)
{
	return (_layerName.size() == 0) ? _channel : _layerName + "." + _channel;
}


bool ImageWriter::write(std::vector<Romanesco::Channel> _channels)
{
	Imf::Header header(_channels[0].m_width, _channels[0].m_height);

	Imf::ChannelList& channels = header.channels();
	Imf::FrameBuffer framebuffer;

	for (unsigned int i = 0; i < _channels.size(); i++)
	{
		Romanesco::Channel& _image = _channels[i];

		std::string name_r = layerChannelString(_image.m_name, "R");
		std::string name_g = layerChannelString(_image.m_name, "G");
		std::string name_b = layerChannelString(_image.m_name, "B");
		std::string name_a = layerChannelString(_image.m_name, "A");

		qDebug() << qPrintable(name_r.c_str()) << qPrintable(name_g.c_str()) << qPrintable(name_b.c_str()) << qPrintable(name_a.c_str());

		channels.insert(name_r, Imf::Channel(Imf::FLOAT));
		channels.insert(name_g, Imf::Channel(Imf::FLOAT));
		channels.insert(name_b, Imf::Channel(Imf::FLOAT));
		channels.insert(name_a, Imf::Channel(Imf::FLOAT));

		char* channel_rPtr = (char*)&(_image.m_pixels[0].r);
		char* channel_gPtr = (char*)&(_image.m_pixels[0].g);
		char* channel_bPtr = (char*)&(_image.m_pixels[0].b);
		char* channel_aPtr = (char*)&(_image.m_pixels[0].a);

		unsigned int xstride = sizeof(half) * 4;
		unsigned int ystride = sizeof(half) * 4 * _image.m_width;

		framebuffer.insert(name_r, Imf::Slice(Imf::FLOAT, channel_rPtr, xstride, ystride));
		framebuffer.insert(name_g, Imf::Slice(Imf::FLOAT, channel_gPtr, xstride, ystride));
		framebuffer.insert(name_b, Imf::Slice(Imf::FLOAT, channel_bPtr, xstride, ystride));
		framebuffer.insert(name_a, Imf::Slice(Imf::FLOAT, channel_aPtr, xstride, ystride));
	}

	Imf::OutputFile file(m_fileName.c_str(), header);
	file.setFrameBuffer(framebuffer);
	file.writePixels(_channels[0].m_height);
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

void ImageWriter::addChannel(/*OpenImageIO::TypeDesc _type,*/ std::string _name)
{
    //// Add the channel name to the spec
    //m_spec->channelnames.push_back(_name);
    //// and it's associated type (this must match the type in the struct we're passing in per pixel)
    //m_spec->channelformats.push_back(_type);
}

void ImageWriter::addChannelRGB(/*OpenImageIO::TypeDesc _type, */std::string _name)
{
    //// Generate names like P.x if one is supplied
    //if(_name.length() > 0)  {
    //    _name = _name + ".";
    //}

    //addChannel(_type, _name + "R");
    //addChannel(_type, _name + "G");
    //addChannel(_type, _name + "B");
}

void ImageWriter::addChannelRGBA(/*OpenImageIO::TypeDesc _type, */std::string _name)
{
   /* if(_name.length() > 0)
    {
        _name = _name + ".";
    }

    addChannel(_type, _name + "R");
    addChannel(_type, _name + "G");
    addChannel(_type, _name + "B");
    addChannel(_type, _name + "A");*/
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