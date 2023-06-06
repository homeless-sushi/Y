#include <Cutcp/Cutcp.h>
#include <Cutcp/CutcpCpu.h>

#include <vector>

#include <Vector/Vec3.h>

#include <Atom/Atom.h>
#include <Atom/Utils.h>

namespace CutcpCpu 
{
    CutcpCpu::CutcpCpu(
                Lattice::Lattice lattice,
                const std::vector<Atom::Atom> atoms,
                float potentialCutoff,
                float exclusionCutoff,
                unsigned int nThreads
    ) : 
        lattice{lattice},
        atoms{atoms},
        potentialCutoff{potentialCutoff},
        exclusionCutoff{exclusionCutoff},
        nThreads{nThreads}
    {};

    CutcpCpu::~CutcpCpu() = default;

    void CutcpCpu::run()
    {
        cutoffPotential();
        removeExclusions();
    }

    Lattice::Lattice CutcpCpu::getResult() { return lattice; };

    void CutcpCpu::cutoffPotential() 
    { 
        const int nx = lattice.nx();
        const int ny = lattice.ny();
        const int nz = lattice.nz();
        const float minx = lattice.min().x;
        const float miny = lattice.min().y;
        const float minz = lattice.min().z;
        const float spacing = lattice.spacing();

        const float cutSqrd = potentialCutoff * potentialCutoff;
        const float inverseCutSqrd = 1.f / cutSqrd;
        const float inverseSpacing = 1.f / spacing;

        // find min and max extent 
        Vector::Vec3 nextMinCoords;
        Vector::Vec3 nextMaxCoords;
        Atom::GetAtomBounds(atoms, nextMinCoords, nextMaxCoords);

        // number of cells in each dimension 
        constexpr float inverseCellLen = INV_CELL_LEN;
        const int nxcell = (int) floor((nextMaxCoords.x-nextMinCoords.x) * inverseCellLen) + 1;
        const int nycell = (int) floor((nextMaxCoords.y-nextMinCoords.y) * inverseCellLen) + 1;
        const int nzcell = (int) floor((nextMaxCoords.z-nextMinCoords.z) * inverseCellLen) + 1;
        const int ncells = nxcell * nycell * nzcell;

        // allocate for cursor link list implementation 
        std::vector<int> fromCellToFirstAtom(ncells, -1);
        std::vector<int> fromAtomToNextAtom(atoms.size(), -1);

        // geometric hashing
        for (unsigned int n = 0;  n < atoms.size();  n++) {

            // skip any non-contributing atoms
            if (atoms[n].q==0) 
            continue;  
            
            const int i = (int) floor((atoms[n].pos.x - nextMinCoords.x) * inverseCellLen);
            const int j = (int) floor((atoms[n].pos.y - nextMinCoords.y) * inverseCellLen);
            const int k = (int) floor((atoms[n].pos.z - nextMinCoords.z) * inverseCellLen);
            const int linearCellIdx = (k*nycell + j)*nxcell + i;

            fromAtomToNextAtom[n] = fromCellToFirstAtom[linearCellIdx];
            fromCellToFirstAtom[linearCellIdx] = n;
        }

        // traverse the grid cells
        //#pragma omp parallel for num_threads(nThreads)
        for (int linearCellIdx = 0;  linearCellIdx < ncells;  linearCellIdx++) {
            for (int atomIdx = fromCellToFirstAtom[linearCellIdx];  atomIdx != -1;  atomIdx = fromAtomToNextAtom[atomIdx]) {
                const float atomX = atoms[atomIdx].pos.x - minx;
                const float atomY = atoms[atomIdx].pos.y - miny;
                const float atomZ = atoms[atomIdx].pos.z - minz;
                const float atomQ = atoms[atomIdx].q;

                // find closest grid point with position less than or equal to atom */
                const int xClosest = (int) (atomX * inverseSpacing);
                const int yClosest = (int) (atomY * inverseSpacing);
                const int zClosest = (int) (atomZ * inverseSpacing);

                // find extent of surrounding box of grid points */
                const int radius = (int) ceilf(potentialCutoff * inverseSpacing) - 1;
                const float ia = std::max(0, xClosest - radius);
                const float ja = std::max(0, yClosest - radius);
                const float ka = std::max(0, zClosest - radius);
                const float ib = std::min(nx-1, xClosest + radius + 1);
                const float jb = std::min(ny-1, yClosest + radius + 1);
                const float kb = std::min(nz-1, zClosest + radius + 1);

                // loop over surrounding grid points */
                float dx, dy, dz;
                float dz2, dydz2, r2;

                const float xStart = ia*spacing - atomX;
                const float yStart = ja*spacing - atomY;
                dz = ka*spacing - atomZ;
                for (int k = ka;  k <= kb;  k++, dz += spacing) {
                    const int koff = k*ny;
                    dz2 = dz*dz;
                    dy = yStart;
                    for (int j = ja;  j <= jb;  j++, dy += spacing) {
                        const int jkoff = (koff + j)*nx;
                        dydz2 = dy*dy + dz2;

                        if (dydz2 >= cutSqrd) 
                            continue;
                    
                        dx = xStart;
                        for (int i = ia;  i <= ib;  i++, dx += spacing) {
                            r2 = dx*dx + dydz2;

                            if (r2 >= cutSqrd)
                                continue;

                            float s = (1.f - r2 * inverseCutSqrd);
                            float e = atomQ * (1/sqrtf(r2)) * s * s;
                            lattice.points[jkoff + i] += e;
                        }
                    }
                } // end loop over surrounding grid points

            } // end loop over atoms in a gridcell
        } // end loop over gridcells
    };

    void CutcpCpu::removeExclusions()
    {
        const int nx = lattice.nx();
        const int ny = lattice.ny();
        const int nz = lattice.nz();
        const float minx = lattice.min().x;
        const float miny = lattice.min().y;
        const float minz = lattice.min().z;
        const float spacing = lattice.spacing();

        const float cutSqrd = exclusionCutoff * exclusionCutoff;
        const float inverseSpacing = 1.f / spacing;

        // find min and max extent 
        Vector::Vec3 nextMinCoords;
        Vector::Vec3 nextMaxCoords;
        Atom::GetAtomBounds(atoms, nextMinCoords, nextMaxCoords);

        // number of cells in each dimension 
        float inverseCellLen = INV_CELL_LEN;
        const int nxcell = (int) floor((nextMaxCoords.x-nextMinCoords.x) * inverseCellLen) + 1;
        const int nycell = (int) floor((nextMaxCoords.y-nextMinCoords.y) * inverseCellLen) + 1;
        const int nzcell = (int) floor((nextMaxCoords.z-nextMinCoords.z) * inverseCellLen) + 1;
        const int ncells = nxcell * nycell * nzcell;

        // geometric hashing 
        std::vector<int> fromCellToFirstAtom(ncells, -1); //cellToFirstAtom[cellIdx] is the first atom in that cell;
        std::vector<int> fromAtomToNextAtom(atoms.size(), -1); //atomToNextCellAtom[atomIdx] is another atom in that cell
        for (unsigned int n = 0; n < atoms.size(); n++) {
        
            if (atoms[n].q == 0) //skip non contributing atoms
                continue;  
            
            const int i = (int) floorf((atoms[n].pos.x - nextMinCoords.x) * inverseCellLen);
            const int j = (int) floorf((atoms[n].pos.y - nextMinCoords.y) * inverseCellLen);
            const int k = (int) floorf((atoms[n].pos.z - nextMinCoords.z) * inverseCellLen);
            const int linearCellIdx = (k*nycell + j)*nxcell + i;

            fromAtomToNextAtom[n] = fromCellToFirstAtom[linearCellIdx];
            fromCellToFirstAtom[linearCellIdx] = n;
        }

        // traverse the grid cells
        //#pragma omp parallel for num_threads(nThreads)
        for (int linearCellIdx = 0;  linearCellIdx < ncells;  linearCellIdx++) {
            for (int atomIdx = fromCellToFirstAtom[linearCellIdx];  atomIdx != -1;  atomIdx = fromAtomToNextAtom[atomIdx]) {
                const float atomX = atoms[atomIdx].pos.x - minx;
                const float atomY = atoms[atomIdx].pos.y - miny;
                const float atomZ = atoms[atomIdx].pos.z - minz;
                const float q = atoms[atomIdx].q;

                // find closest grid point with position less than or equal to atom */
                const int xClosest = (int) (atomX * inverseSpacing);
                const int yClosest = (int) (atomY * inverseSpacing);
                const int zClosest = (int) (atomZ * inverseSpacing);

                // trim the cell search space according to the cutoff
                const int radius = (int) ceilf(exclusionCutoff * inverseSpacing) - 1; // radius as a number of cells
                const int ia = std::max(0, xClosest - radius);
                const int ja = std::max(0, yClosest - radius);
                const int ka = std::max(0, zClosest - radius);
                const int ib = std::min(nx-1, xClosest + radius + 1);
                const int jb = std::min(ny-1, yClosest + radius + 1);
                const int kb = std::min(nz-1, zClosest + radius + 1);

                // loop over surrounding grid points
                float dx, dy, dz;
                float dz2, dydz2, r2;

                const float xStart = ia*spacing - atomX; // distance between fist cell's X and atomX
                const float yStart = ja*spacing - atomY; // distance between fist cell's Y and atomY
                dz = ka*spacing - atomZ;
                for (int k = ka;  k <= kb;  k++, dz += spacing) {
                    const int koff = k*ny;
                    dz2 = dz*dz;

                    dy = yStart;
                    for (int j = ja;  j <= jb;  j++, dy += spacing) {
                        const int jkoff = (koff + j)*nx;
                        dydz2 = dy*dy + dz2;

                        dx = xStart;
                        for (int i = ia;  i <= ib;  i++, dx += spacing) {
                            r2 = dx*dx + dydz2;

                            // If atom and lattice point are too close, set the lattice value
                            if (r2 < cutSqrd)
                                lattice.points[jkoff + i] = 0;
                        }
                    }
                } // for each grid point in the radius

            } // for each atom in a cell
        } // for each cell
    };    
}
