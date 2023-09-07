#include <Cutcp/Lattice.h>
#include <Cutcp/Cutcp.h>
#include <Cutcp/CutcpCuda.h>

#include <map>
#include <vector>

#include <cuda_runtime.h>

#include <Atom/Atom.h>
#include <Atom/Utils.h>

#include "CudaError/CudaError.h"

namespace CutcpCuda
{
    AtomsCrs::AtomsCrs(
        const Lattice::Lattice& lattice,
        const std::vector<Atom::Atom>& atoms
    ) :
        owner{true}
    {

        constexpr float inverseCellLen = INV_CELL_LEN;
        xNCells = (int) floor((lattice.max().x-lattice.min().x) * inverseCellLen) + 1;
        yNCells = (int) floor((lattice.max().y-lattice.min().y) * inverseCellLen) + 1;
        zNCells = (int) floor((lattice.max().z-lattice.min().z) * inverseCellLen) + 1;
        nCells = xNCells * yNCells * zNCells;
        
        std::map<int, std::vector<Atom::Atom>> atomsMap;
        long nAtoms = 0;
        for (const Atom::Atom& atom : atoms) {

            //skip any non-contributing atoms
            if (atom.q==0) 
                continue;  
            
            const int i = (int) floor((atom.pos.x - lattice.min().x) * inverseCellLen);
            const int j = (int) floor((atom.pos.y - lattice.min().y) * inverseCellLen);
            const int k = (int) floor((atom.pos.z - lattice.min().z) * inverseCellLen);
            const int linearCellIdx = (i*yNCells + j)*zNCells + k;

            if (atomsMap.find(linearCellIdx) == atomsMap.end())
                atomsMap.emplace(std::make_pair(linearCellIdx, 0));

            atomsMap[linearCellIdx].push_back(atom);
            nAtoms++;
        }
        
        std::vector<Atom::Atom> atomsValuesHost;
        std::vector<long> cellIndexesHost(nCells+1, 0);
        long currCellIndex = 0;
        for(long i = 0; i < nCells; i++){

            if (atomsMap.find(i) != atomsMap.end()){
                currCellIndex += atomsMap[i].size();
                atomsValuesHost.insert(atomsValuesHost.end(), atomsMap[i].begin(), atomsMap[i].end());
            }

            cellIndexesHost[i+1] = currCellIndex;
        }

        CudaErrorCheck(cudaMalloc(&atomsValues, sizeof(Atom::Atom)*atomsValuesHost.size()));
        CudaErrorCheck(cudaMalloc(&cellIndexes, sizeof(long)*cellIndexesHost.size()));
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(atomsValues, atomsValuesHost.data(), sizeof(Atom::Atom)*atomsValuesHost.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(cellIndexes, cellIndexesHost.data(), sizeof(long)*cellIndexesHost.size(), cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dataUploadTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    AtomsCrs::AtomsCrs(const AtomsCrs& owner) :
        owner{false},
        atomsValues{owner.atomsValues},
        nAtoms{owner.nAtoms},
        cellIndexes{owner.cellIndexes},
        xNCells{owner.xNCells},
        yNCells{owner.yNCells},
        zNCells{owner.zNCells},
        nCells{owner.nCells},
        dataUploadTime{owner.dataUploadTime}
    {};

    AtomsCrs::~AtomsCrs()
    {   
        nAtoms = 0;
        if(owner){
            CudaErrorCheck(cudaFree(atomsValues));
            CudaErrorCheck(cudaFree(cellIndexes));
        }
        atomsValues = nullptr;
        cellIndexes = nullptr;
    }

    LatticeCuda::LatticeCuda(
        const Lattice::Lattice& lattice,
        const std::vector<Atom::Atom>& atoms
    ) :
        owner{true},
        nxPoints{lattice.nx()},
        nyPoints{lattice.ny()},
        nzPoints{lattice.nz()},
        nPoints{lattice.n()},
        spacing{lattice.spacing()},
        min{lattice.min()},
        max{lattice.max()}
    {
        CudaErrorCheck(cudaMalloc(&points, sizeof(float)*nPoints));
        CudaErrorCheck(cudaMemset(points, 0.f, sizeof(float)*nPoints));
    };

    LatticeCuda::LatticeCuda(const LatticeCuda& owner) :
        owner{false},
        nxPoints{owner.nxPoints},
        nyPoints{owner.nyPoints},
        nzPoints{owner.nzPoints},
        nPoints{owner.nPoints},
        spacing{owner.spacing},
        min{owner.min},
        max{owner.max},
        points{owner.points}
    {};

    LatticeCuda::~LatticeCuda()
    {   
        nxPoints = 0;
        nyPoints = 0;
        nzPoints = 0;
        nPoints = 0;
        if(owner){
            CudaErrorCheck(cudaFree(points));
        }
        points = nullptr;
    }

    CutcpCuda::CutcpCuda(
        const Lattice::Lattice& lattice,
        const std::vector<Atom::Atom>& atoms,
        float potentialCutoff,
        float exclusionCutoff,
        unsigned int blockSize
    ) :
        lattice{lattice},
        atoms{atoms},
        potentialCutoff{potentialCutoff},
        exclusionCutoff{exclusionCutoff},
        blockSize{blockSize},
        latticeCuda{lattice,atoms},
        atomCrs{lattice,atoms},
        dataUploadTime{atomCrs.getDataUploadTime()}
    {};

    CutcpCuda::~CutcpCuda(){};

    Lattice::Lattice CutcpCuda::getResult(){
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(
            lattice.points.data(),
            latticeCuda.points,
            sizeof(float)*lattice.points.size(),
            cudaMemcpyDeviceToHost
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dataDownloadTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return lattice;
    };
    
    __global__
    void cutoffPotential(
        AtomsCrs atoms,
        LatticeCuda lattice,
        float potentialCutoff
    ) 
    {
        extern __shared__ float e[];

        const float cutoffSqrd = potentialCutoff*potentialCutoff;
        for(long pointIdx = blockIdx.x; pointIdx < lattice.nPoints; pointIdx+=gridDim.x){

            const long xPointIdx = pointIdx/(lattice.nyPoints*lattice.nzPoints);
            const long xPointRemainder = pointIdx%(lattice.nyPoints*lattice.nzPoints);
            const long yPointIdx = xPointRemainder/lattice.nzPoints;
            const long zPointIdx = xPointRemainder%lattice.nzPoints;

            const float xPointCoord = xPointIdx * lattice.spacing;
            const float yPointCoord = yPointIdx * lattice.spacing;
            const float zPointCoord = zPointIdx * lattice.spacing;

            const long xCellIdx = (int) floor(xPointCoord * INV_CELL_LEN);
            const long yCellIdx = (int) floor(yPointCoord * INV_CELL_LEN);
            const long zCellIdx = (int) floor(zPointCoord * INV_CELL_LEN);

            const long cellRadius = ceil(potentialCutoff*INV_CELL_LEN);
            const long xNeighbourStart = max(0L, xCellIdx - cellRadius);
            const long xNeighbourStop = min(atoms.xNCells, xCellIdx + cellRadius + 1);
            const long yNeighbourStart = max(0L, yCellIdx - cellRadius);
            const long yNeighbourStop = min(atoms.yNCells, yCellIdx + cellRadius + 1);
            const long zNeighbourStart = max(0L, zCellIdx - cellRadius);
            const long zNeighbourStop = min(atoms.zNCells, zCellIdx + cellRadius + 1);

            const float xPointAbsCoord = xPointCoord + lattice.min.x;
            const float yPointAbsCoord = yPointCoord + lattice.min.y;
            const float zPointAbsCoord = zPointCoord + lattice.min.z;

            e[threadIdx.x]=0;
            for(long xNeighbour = xNeighbourStart; xNeighbour < xNeighbourStop; xNeighbour++){
                for(long yNeighbour = yNeighbourStart; yNeighbour < yNeighbourStop; yNeighbour++){

                    long cellLinearOffset = (xNeighbour*atoms.yNCells + yNeighbour)*atoms.zNCells;
                    long cellLinearIndexStart = cellLinearOffset + zNeighbourStart;
                    long cellLinearIndexStop = cellLinearOffset + zNeighbourStop;
                    long atomIdxStart = atoms.cellIndexes[cellLinearIndexStart];
                    long atomIdxStop = atoms.cellIndexes[cellLinearIndexStop];

                    for(long atomIdx = atomIdxStart+threadIdx.x; atomIdx < atomIdxStop; atomIdx+=blockDim.x){

                        Atom::Atom atom = atoms.atomsValues[atomIdx];
                        const float dx = atom.pos.x - xPointAbsCoord;
                        const float dy = atom.pos.y - yPointAbsCoord;
                        const float dz = atom.pos.z - zPointAbsCoord;
                        const float r2 = dx*dx+dy*dy+dz*dz;

                        if (r2 < cutoffSqrd){
                            const float inverseCutSqrd = 1/cutoffSqrd;
                            const float s = (1.f - r2 * inverseCutSqrd);
                            e[threadIdx.x] += atom.q * (rsqrt(r2)) * s * s;
                        }
                    }
                }
            }

            __syncthreads();
            for(unsigned int i = blockDim.x>>1; i > 0; i>>=1){
                if(threadIdx.x < i)
                    e[threadIdx.x]+=e[threadIdx.x+i];
                __syncthreads();
            }
            if(threadIdx.x == 0)
                lattice.points[pointIdx] = e[threadIdx.x];
            __syncthreads();
        }
    };

    __global__
    void cutoffExclusion(
        AtomsCrs atoms,
        LatticeCuda lattice,
        float exclusionCutoff
    ) 
    {
        __shared__ bool exclude[1];

        const float cutoffSqrd = exclusionCutoff*exclusionCutoff;
        for(long pointIdx = blockIdx.x; pointIdx < lattice.nPoints; pointIdx+=gridDim.x){

            if(threadIdx.x == 0)
                exclude[0] = false;

            const long xPointIdx = pointIdx/(lattice.nyPoints*lattice.nzPoints);
            const long xPointRemainder = pointIdx%(lattice.nyPoints*lattice.nzPoints);
            const long yPointIdx = xPointRemainder/lattice.nzPoints;
            const long zPointIdx = xPointRemainder%lattice.nzPoints;

            const float xPointCoord = xPointIdx * lattice.spacing;
            const float yPointCoord = yPointIdx * lattice.spacing;
            const float zPointCoord = zPointIdx * lattice.spacing;

            const long xCellIdx = (int) floor(xPointCoord * INV_CELL_LEN);
            const long yCellIdx = (int) floor(yPointCoord * INV_CELL_LEN);
            const long zCellIdx = (int) floor(zPointCoord * INV_CELL_LEN);

            const long cellRadius = ceil(exclusionCutoff*INV_CELL_LEN);
            const long xNeighbourStart = max(0L, xCellIdx - cellRadius);
            const long xNeighbourStop = min(atoms.xNCells, xCellIdx + cellRadius + 1);
            const long yNeighbourStart = max(0L, yCellIdx - cellRadius);
            const long yNeighbourStop = min(atoms.yNCells, yCellIdx + cellRadius + 1);
            const long zNeighbourStart = max(0L, zCellIdx - cellRadius);
            const long zNeighbourStop = min(atoms.zNCells, zCellIdx + cellRadius + 1);

            const float xPointAbsCoord = xPointCoord + lattice.min.x;
            const float yPointAbsCoord = yPointCoord + lattice.min.y;
            const float zPointAbsCoord = zPointCoord + lattice.min.z;

            __syncthreads();
            for(long xNeighbour = xNeighbourStart; xNeighbour < xNeighbourStop && !exclude[0]; xNeighbour++){
                for(long yNeighbour = yNeighbourStart; yNeighbour < yNeighbourStop && !exclude[0]; yNeighbour++){

                    long cellLinearOffset = (xNeighbour*atoms.yNCells + yNeighbour)*atoms.zNCells;
                    long cellLinearIndexStart = cellLinearOffset + zNeighbourStart;
                    long cellLinearIndexStop = cellLinearOffset + zNeighbourStop;
                    long atomIdxStart = atoms.cellIndexes[cellLinearIndexStart];
                    long atomIdxStop = atoms.cellIndexes[cellLinearIndexStop];

                    for(long atomIdx = atomIdxStart+threadIdx.x; atomIdx < atomIdxStop && !exclude[0]; atomIdx+=blockDim.x){

                        Atom::Atom atom = atoms.atomsValues[atomIdx];
                        const float dx = atom.pos.x - xPointAbsCoord;
                        const float dy = atom.pos.y - yPointAbsCoord;
                        const float dz = atom.pos.z - zPointAbsCoord;
                        const float r2 = dx*dx+dy*dy+dz*dz;

                        if(r2 < cutoffSqrd)
                            exclude[0] = true;
                    }
                }
            }
            __syncthreads();
            if(threadIdx.x == 0 && exclude[0])
                lattice.points[pointIdx] = 0;
            __syncthreads();
        }
    };

    void CutcpCuda::run()
    {   
        cudaDeviceProp deviceProp;
        CudaErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));
        unsigned int smCount = deviceProp.multiProcessorCount;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cutoffPotential<<<smCount,blockSize,blockSize*sizeof(float)>>>(
            atomCrs,
            latticeCuda,
            potentialCutoff
        );
        cutoffExclusion<<<smCount,blockSize,sizeof(bool)>>>(
            atomCrs,
            latticeCuda,
            exclusionCutoff
        );
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&kernelTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };
}