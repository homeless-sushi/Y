#ifndef CUTCP_CUTCP_CUTCP_CPU_H
#define CUTCP_CUTCP_CUTCP_CPU_H

#include <Cutcp/Lattice.h>
#include <Cutcp/Cutcp.h>

#include <vector>

#include <Atom/Atom.h>

namespace CutcpCpu
{
    class CutcpCpu : public Cutcp::Cutcp 
    {
        public:
            CutcpCpu(
                Lattice::Lattice lattice,
                const std::vector<Atom::Atom> atoms,
                float potentialCutoff,
                float exclusionCutoff,
                unsigned int nThreads
            );
            ~CutcpCpu() override;
            
            void run() override;
            Lattice::Lattice getResult() override;

        private:
            Lattice::Lattice lattice;
            const std::vector<Atom::Atom> atoms;
            float potentialCutoff;
            float exclusionCutoff;

            unsigned int nThreads;

            void cutoffPotential(); //first kernel
            void removeExclusions(); //second kernel

    };
}

#endif //CUTCP_CUTCP_CUTCP_CPU_H
