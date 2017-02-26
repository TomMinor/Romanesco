#ifndef IMAGE_WRITER__
#define IMAGE_WRITER__

#include <vector>

#include <OpenImageIO/imageio.h>
#include <OpenEXR/half.h>


class ImageWriter
{
public:
    struct Pixel
    {
        ///
        /// \brief Default channels, we'll always need these
        ///
        half r, g, b, a;

        ///
        /// \brief z depth
        ///
        half z;

        ///
        /// \brief trapR 3 orbit trap channels
        ///
        half trapR, trapB, trapG;

        ///
        /// \brief iteration count
        ///
        half iteration;

        ///
        /// \brief x_nrm
        ///
        half x_nrm, y_nrm, z_nrm;

        ///
        /// \brief x_pos
        ///y
        half x_pos, y_pos, z_pos;

        ///
        /// \brief diffuseR
        ///
        half diffuseR, diffuseG, diffuseB;

    };

    ImageWriter(std::string _filename, unsigned int _width, unsigned int _height);
    ~ImageWriter();

    bool write(std::vector<Pixel> _pixels);

    std::string getFileName() { return m_fileName; }

private:
    static bool progressCallback(void* _data, float _progress);

    ///
    /// \brief addChannel Add a single channel to the image spec
    /// \param _name
    /// \param _type
    ///
    void addChannel(OpenImageIO::TypeDesc _type, std::string _name);

    ///
    /// \brief addChannelRGB Adds a channel to the image spec, using the special syntax that designates related channels (P.x, P.y, P.z) if a channel name is passed
    /// \param _name
    /// \param _type
    ///
    void addChannelRGB(OpenImageIO::TypeDesc _type, std::string _name = "");

    ///
    /// \brief addChannelRGBA Adds a channel to the image spec, using the special syntax that designates related channels (P.x, P.y, P.z, P.a) if a channel name is passed
    /// \param _name
    /// \param _type
    ///
    void addChannelRGBA(OpenImageIO::TypeDesc _type, std::string _name = "");

private:
    std::string m_fileName;

    OpenImageIO::ImageOutput *m_outFile;
    OpenImageIO::ImageSpec *m_spec;
};

#endif
