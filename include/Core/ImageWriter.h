#ifndef IMAGE_WRITER__
#define IMAGE_WRITER__

#include <vector>

#ifdef __WIN32
#undef max
#undef min
#endif
///@todo Why does this have to be defined?
#define OPENEXR_DLL
#include <OpenEXR/ImfRgba.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/half.h>
#undef OPENEXR_DLL

namespace Romanesco
{
	struct Channel
	{
	public:
		Channel(float* _pixels, unsigned int _width, unsigned int _height, std::string _name = "")
			: m_width(_width), m_height(_height), m_name(_name)
		{
			m_pixels = new Imf::Rgba[m_width * m_height];
			std::fill(m_pixels, m_pixels + (m_width * m_height), Imf::Rgba(1.f, 1.f, 1.f, 1.f));

			for (int i = 0; i < 4 * m_width * m_height; i += 4)
			{
				//unsigned int idx = i + (j * m_width);

				float R = _pixels[i];
				float G = _pixels[i + 1];
				float B = _pixels[i + 2];
				float A = _pixels[i + 3];

				//setPixel(i, j, Imf::Rgba(R, G, B, A) );
				m_pixels[i / 4] = Imf::Rgba(R, G, B, A);
			}
		}

		void setPixel(int x, int y, Imf::Rgba _val)
		{
			m_pixels[x + (y * m_width)] = _val;
		}

		~Channel()
		{
			//delete m_pixels;
		}

		//private:
		Imf::Rgba* m_pixels;
		unsigned int m_width, m_height;
		std::string m_name;
	};
}

class ImageWriter
{
public:
    ImageWriter(std::string _filename, unsigned int _width, unsigned int _height);
    ~ImageWriter();

	bool write(std::vector<Romanesco::Channel> _channels);

    std::string getFileName() { return m_fileName; }

private:
    static bool progressCallback(void* _data, float _progress);

    ///
    /// \brief addChannel Add a single channel to the image spec
    /// \param _name
    /// \param _type
    ///
    void addChannel(/*OpenImageIO::TypeDesc _type,*/ std::string _name);

    ///
    /// \brief addChannelRGB Adds a channel to the image spec, using the special syntax that designates related channels (P.x, P.y, P.z) if a channel name is passed
    /// \param _name
    /// \param _type
    ///
    void addChannelRGB(/*OpenImageIO::TypeDesc _type,*/ std::string _name = "");

    ///
    /// \brief addChannelRGBA Adds a channel to the image spec, using the special syntax that designates related channels (P.x, P.y, P.z, P.a) if a channel name is passed
    /// \param _name
    /// \param _type
    ///
    void addChannelRGBA(/*OpenImageIO::TypeDesc _type,*/ std::string _name = "");

private:
    std::string m_fileName;

    //OpenImageIO::ImageOutput *m_outFile;
    //OpenImageIO::ImageSpec *m_spec;
};

#endif
