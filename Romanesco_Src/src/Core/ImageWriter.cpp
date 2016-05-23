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
    : m_outFile(nullptr), m_spec(nullptr), m_fileName(_filename)
{
    m_spec = new OpenImageIO::ImageSpec(_width, _height, 11, OpenImageIO::TypeDesc::HALF);
    m_spec->channelnames.clear();
    m_spec->attribute("compression", "zip");

    // Channel setup
    addChannelRGBA(OpenImageIO::TypeDesc::HALF);        // RGBA Channels
    addChannel(OpenImageIO::TypeDesc::DOUBLE, "Z");     // Depth Channel
    addChannelRGB(OpenImageIO::TypeDesc::HALF, "N");    // Normal Channels
    addChannelRGB(OpenImageIO::TypeDesc::HALF, "P");    // World Position Channels

    // Tell OIIO that these specific channels will store alpha/depth (helps for some formats)
    m_spec->alpha_channel = 3;
    m_spec->z_channel = 4;

    // Handy for debugging later?
    m_spec->attribute("Romanesco Version", 0.1f);
}

ImageWriter::~ImageWriter()
{
    if(m_outFile)
    {
        m_outFile->close();
//        OpenImageIO::ImageOutput::destroy(m_outFile);

        m_outFile = nullptr;
    }

    if(m_spec)
    {
        delete m_spec;
    }
}

bool ImageWriter::write(std::vector<ImageWriter::Pixel> _pixels)
{
    m_outFile = nullptr;
    m_outFile = OpenImageIO::ImageOutput::create(m_fileName);

    if(!m_outFile)
    {
        return false;
    }

    OpenImageIO::ProgressCallback callback = ImageWriter::progressCallback;
    m_outFile->open(m_fileName, *m_spec);

    bool success = false;

    // Double check that this format accepts per-channel formats
    if (!m_outFile->supports("channelformats")) {
        qDebug("Warning: Filetype %s doesn't support multichannel data", m_outFile->format_name());

//        OpenImageIO::ImageOutput::destroy(m_outFile);
        m_outFile->close();
        throw std::runtime_error("Can't write per channel data (passes) in ImageWriter");
    }

    success = m_outFile->write_image (OpenImageIO::TypeDesc::UNKNOWN,
                            _pixels.data(),
                            sizeof( ImageWriter::Pixel ),
                            OpenImageIO::AutoStride,
                            OpenImageIO::AutoStride,
                            callback,
                            this // This class is the callback payload
                            );

    m_outFile->close();


    return success;
}

void ImageWriter::addChannel(OpenImageIO::TypeDesc _type, std::string _name)
{
    // Add the channel name to the spec
    m_spec->channelnames.push_back(_name);
    // and it's associated type (this must match the type in the struct we're passing in per pixel)
    m_spec->channelformats.push_back(_type);
}

void ImageWriter::addChannelRGB(OpenImageIO::TypeDesc _type, std::string _name)
{
    // Generate names like P.x if one is supplied
    if(_name.length() > 0)  {
        _name = _name + ".";
    }

    addChannel(_type, _name + "R");
    addChannel(_type, _name + "G");
    addChannel(_type, _name + "B");
}

void ImageWriter::addChannelRGBA(OpenImageIO::TypeDesc _type, std::string _name)
{
    if(_name.length() > 0)
    {
        _name = _name + ".";
    }

    addChannel(_type, _name + "R");
    addChannel(_type, _name + "G");
    addChannel(_type, _name + "B");
    addChannel(_type, _name + "A");
}
