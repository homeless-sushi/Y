#include <Sgemm/Matrix.h>
#include <Sgemm/Sgemm.h>

#include <sstream>
#include <stdexcept>

namespace Sgemm
{
    Sgemm::Sgemm(
        float alpha,
        float beta,
        Matrix& a,
        Matrix& b,
        Matrix& c
    ) :
        alpha_{alpha},
        beta_{beta},
        a_{a},
        b_{b},
        c_{c},
        res_{c.nrows(), c.ncols()}
    {
        if(
            (a.ncols() != b.nrows()) ||
            (a.nrows() != c.nrows()) ||
            (b.ncols() != c.ncols())
        ){
            std::ostringstream errorMsg;
            errorMsg << "Matrix dimensions mismatch:" << "\n"
                << "\tA is " << a.nrows() << "x"<< a.ncols() << "\n"
                << "\tB is " << b.nrows() << "x"<< b.ncols() << "\n"
                << "\tC is " << c.nrows() << "x"<< c.ncols() << "\n";
            throw std::runtime_error(errorMsg.str());
        }
    };
}
