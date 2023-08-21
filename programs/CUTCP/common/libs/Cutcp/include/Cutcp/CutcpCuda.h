#ifndef CUTCP_CUTCP_CUTCP_CUDA_H
#define CUTCP_CUTCP_CUTCP_CUDA_H

#include <Cutcp/Lattice.h>
#include <Cutcp/Cutcp.h>

#include <vector>

#include <Atom/Atom.h>

#include <Vector/Vec3.h>

namespace CutcpCuda
{
    class AtomsCrs 
    {
        public: 
            Atom::Atom* atomsValues;
            long nAtoms;

            long* cellIndexes;
            long xNCells;
            long yNCells;
            long zNCells;
            long nCells;

            AtomsCrs(
                const Lattice::Lattice& lattice,
                const std::vector<Atom::Atom>& atoms
            );

            AtomsCrs(const AtomsCrs& owner);
            AtomsCrs& operator=(const AtomsCrs& other) = delete;

            AtomsCrs(AtomsCrs&& other) = delete;
            AtomsCrs& operator=(AtomsCrs&& other) = delete;

            ~AtomsCrs();

            void swap(AtomsCrs& other) {
                std::swap(atomsValues, other.atomsValues);
                std::swap(nAtoms, other.nAtoms);
                std::swap(cellIndexes, other.cellIndexes);
                
                std::swap(xNCells, other.xNCells);
                std::swap(yNCells, other.yNCells);
                std::swap(zNCells, other.zNCells);
                std::swap(nCells, other.nCells);
                std::swap(owner, other.owner);

                std::swap(dataUploadTime, other.dataUploadTime);
            }

            float getDataUploadTime() { return dataUploadTime; };

        private:
            bool owner;

            float dataUploadTime;
    };

    class LatticeCuda 
    {
        public: 
            long nxPoints;
            long nyPoints;
            long nzPoints;
            long nPoints;
            float* points;

            float spacing;

            Vector::Vec3 min;
            Vector::Vec3 max;

            LatticeCuda(
                const Lattice::Lattice& lattice,
                const std::vector<Atom::Atom>& atoms
            );

            LatticeCuda(const LatticeCuda& owner);
            LatticeCuda& operator=(const LatticeCuda& other) = delete;

            LatticeCuda(LatticeCuda&& other) = delete;
            LatticeCuda& operator=(LatticeCuda&& other) = delete;

            ~LatticeCuda();


            void swap(LatticeCuda& other) {
                std::swap(nxPoints, other.nxPoints);
                std::swap(nyPoints, other.nyPoints);
                std::swap(nzPoints, other.nzPoints);
                std::swap(nPoints, other.nPoints);
                std::swap(points, other.points);
                std::swap(spacing, other.spacing);
                std::swap(min, other.min);
                std::swap(max, other.max);
            };

        private:
            bool owner;
    };

    class CutcpCuda : public Cutcp::Cutcp 
    {
        public:
            CutcpCuda(
                const Lattice::Lattice& lattice,
                const std::vector<Atom::Atom>& atoms,
                float potentialCutoff,
                float exclusionCutoff,
                unsigned int blockSize
            );
            ~CutcpCuda() override;
            
            void run() override;
            Lattice::Lattice getResult() override;

            float getDataUploadTime() { return dataUploadTime; }
            float getKernelTime() { return kernelTime; }
            float getDataDownloadTime() { return dataDownloadTime; }

        private:
            Lattice::Lattice lattice;
            const std::vector<Atom::Atom> atoms;
            float potentialCutoff;
            float exclusionCutoff;

            LatticeCuda latticeCuda;
            AtomsCrs atomCrs;

            unsigned int blockSize;

            float dataUploadTime = 0;
            float kernelTime = 0;
            float dataDownloadTime = 0;
    };
}

#endif //CUTCP_CUTCP_CUTCP_CUDA_H
