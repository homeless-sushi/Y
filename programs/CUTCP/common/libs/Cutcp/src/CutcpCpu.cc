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

        // number of cells in each dimension 
        constexpr float inverseCellLen = INV_CELL_LEN;
        const int nxcell = (int) floor((lattice.max().x-lattice.min().x) * inverseCellLen) + 1;
        const int nycell = (int) floor((lattice.max().y-lattice.min().y) * inverseCellLen) + 1;
        const int nzcell = (int) floor((lattice.max().z-lattice.min().z) * inverseCellLen) + 1;
        const int ncells = nxcell * nycell * nzcell;

        // allocate for cursor link list implementation 
        std::vector<int> fromCellToFirstAtom(ncells, -1);
        std::vector<int> fromAtomToNextAtom(atoms.size(), -1);

        // geometric hashing
        for (unsigned int n = 0;  n < atoms.size();  n++) {

            // skip any non-contributing atoms
            if (atoms[n].q==0) 
            continue;  
            
            const int i = (int) floor((atoms[n].pos.x - lattice.min().x) * inverseCellLen);
            const int j = (int) floor((atoms[n].pos.y - lattice.min().y) * inverseCellLen);
            const int k = (int) floor((atoms[n].pos.z - lattice.min().z) * inverseCellLen);
            const int linearCellIdx = (i*nycell + j)*nzcell + k;

            fromAtomToNextAtom[n] = fromCellToFirstAtom[linearCellIdx];
            fromCellToFirstAtom[linearCellIdx] = n;
        }

        // traverse the grid cells
        #pragma omp parallel num_threads(nThreads)
        {
            std::vector<float> localPoints(lattice.points.size(), 0);
            
            #pragma omp for nowait 
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
                    const float iStart = std::max(0, xClosest - radius);
                    const float jStart = std::max(0, yClosest - radius);
                    const float kStart = std::max(0, zClosest - radius);
                    const float iStop = std::min(nx-1, xClosest + radius + 1);
                    const float jStop = std::min(ny-1, yClosest + radius + 1);
                    const float kStop = std::min(nz-1, zClosest + radius + 1);

                    float dx = iStart*spacing - atomX;
                    const float yStart = jStart*spacing - atomY;
                    const float zStart = kStart*spacing - atomZ;
                    for (int i = iStart; i <= iStop; i++, dx+=spacing) {

                        const int iOffset = i*ny;
                        float dx2 = dx*dx;
                        float dy = yStart;

                        for (int j = jStart; j <= jStop; j++, dy+=spacing) {

                            const int ijOffset = (iOffset + j)*nz;
                            float dxdy2 = dx2+ dy*dy;

                            if (dxdy2 >= cutSqrd) 
                                continue;
                        
                            float dz = zStart;
                            for (int k = kStart; k <= kStop; k++, dz+=spacing) {

                                float r2 = dxdy2 + dz*dz;

                                if (r2 >= cutSqrd)
                                    continue;

                                float s = (1.f - r2 * inverseCutSqrd);
                                float e = atomQ * (1/sqrtf(r2)) * s * s;
                                localPoints[ijOffset + k] += e;
                            }
                        }
                    } // end loop over surrounding grid points

                } // end loop over atoms in a gridcell
            } // end loop over gridcells

            #pragma omp critical(ADD_POINTS)
            {
                const unsigned int n = localPoints.size();
                for(unsigned int i = 0; i < n; ++i){
                        lattice.points[i]+=localPoints[i];
                }
            }
        }
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

        // number of cells in each dimension 
        float inverseCellLen = INV_CELL_LEN;
        const int nxcell = (int) floor((lattice.max().x-lattice.min().x) * inverseCellLen) + 1;
        const int nycell = (int) floor((lattice.max().y-lattice.min().y) * inverseCellLen) + 1;
        const int nzcell = (int) floor((lattice.max().z-lattice.min().z) * inverseCellLen) + 1;
        const int ncells = nxcell * nycell * nzcell;

        // geometric hashing 
        std::vector<int> fromCellToFirstAtom(ncells, -1); //cellToFirstAtom[cellIdx] is the first atom in that cell;
        std::vector<int> fromAtomToNextAtom(atoms.size(), -1); //atomToNextCellAtom[atomIdx] is another atom in that cell
        for (unsigned int n = 0; n < atoms.size(); n++) {
        
            if (atoms[n].q == 0) //skip non contributing atoms
                continue;  
            
            const int i = (int) floorf((atoms[n].pos.x - lattice.min().x) * inverseCellLen);
            const int j = (int) floorf((atoms[n].pos.y - lattice.min().y) * inverseCellLen);
            const int k = (int) floorf((atoms[n].pos.z - lattice.min().z) * inverseCellLen);
            const int linearCellIdx = (i*nycell + j)*nzcell + k;

            fromAtomToNextAtom[n] = fromCellToFirstAtom[linearCellIdx];
            fromCellToFirstAtom[linearCellIdx] = n;
        }

        // traverse the grid cells
        #pragma omp parallel for num_threads(nThreads)
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
                const int iStart = std::max(0, xClosest - radius);
                const int jStart = std::max(0, yClosest - radius);
                const int kStart = std::max(0, zClosest - radius);
                const int iStop = std::min(nx-1, xClosest + radius + 1);
                const int jStop = std::min(ny-1, yClosest + radius + 1);
                const int kStop = std::min(nz-1, zClosest + radius + 1);

                // loop over surrounding grid points
                float dx = iStart*spacing - atomX;
                const float yStart = jStart*spacing - atomY;
                const float zStart = kStart*spacing - atomZ;
                for (int i = iStart; i <= iStop; i++, dx += spacing) {

                    const int iOffset = i*ny;
                    float dx2 = dx*dx;
                    float dy = yStart;

                    for (int j = jStart; j <= jStop; j++, dy += spacing) {

                        const int ijOffset = (iOffset + j)*nz;
                        float dxdy2 = dy*dy + dx2;
                    
                        float dz = zStart;
                        for (int k = kStart; k <= kStop; k++, dz += spacing) {

                            float r2 = dxdy2 + dz*dz;

                            if (r2 < cutSqrd)
                                lattice.points[ijOffset + k] = 0;
                        }
                    }
                } // for each grid point in the radius

            } // for each atom in a cell
        } // for each cell
    };    
}
