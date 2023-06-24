#ifndef up_down_pass_h
#define up_down_pass_h
#include "logger.h"
#include "namespace.h"
#include "types.h"

namespace EXAFMM_NAMESPACE {
  class UpDownPass {
  private:
    Kernel & kernel;                                            //!< Kernel class

  private:
    //! Post-order traversal for upward pass
    void postOrderTraversal(GC_iter C, GC_iter C0) {
      int ichild = C->*(static_cast<int Cell::*>(&CellBase::ICHILD));
      int nchild = C->*(static_cast<int Cell::*>(&CellBase::NCHILD));

      my_ityr::parallel_for<my_ityr::access_mode::read>(
          ityr::count_iterator<int>(0),
          ityr::count_iterator<int>(nchild),
          [=](int i) {
        postOrderTraversal(C0 + ichild + i, C0);
      });

      if(nchild==0) {
        my_ityr::with_checkout_tied<my_ityr::access_mode::read>(
            C, 1, [&](const Cell* C_) {
          kernel.P2M(C_);                           // P2M kernel
        });
      } else {                                                    // If not leaf cell
        my_ityr::with_checkout_tied<my_ityr::access_mode::read>(
            C, 1, [&](const Cell* C_) {
          // TODO: refactor
          my_ityr::with_checkout_tied<my_ityr::access_mode::read>(
              C0 + C_->ICHILD, C_->NCHILD, [&](const Cell* Cj0_) {
            kernel.M2M(C_, Cj0_);                                      //  M2M kernel
          });
        });
      }                                                         // End if for non leaf cell
    };

    //! Pre-order traversal for downward pass
    void preOrderTraversal(GC_iter C, GC_iter C0) {
      int ichild = C->*(static_cast<int Cell::*>(&CellBase::ICHILD));
      int nchild = C->*(static_cast<int Cell::*>(&CellBase::NCHILD));
      my_ityr::with_checkout_tied<my_ityr::access_mode::read>(
          C, 1, [&](const Cell* C_) {
        // TODO: refactor
        my_ityr::with_checkout_tied<my_ityr::access_mode::read>(
            C0 + C_->IPARENT, 1, [&](const Cell* Cj0_) {
          kernel.L2L(C_, Cj0_);                                        //  L2L kernel
        });
      });
      if (nchild==0) {                                       //  If leaf cell
        my_ityr::with_checkout_tied<my_ityr::access_mode::read>(
            C, 1, [&](const Cell* C_) {
          kernel.L2P(C_);                                          //  L2P kernel
        });
      }                                                         // End if for leaf cell
#if 0
#if EXAFMM_USE_WEIGHT
      C_iter CP = C0 + C->IPARENT;                              // Parent cell
      C->WEIGHT += CP->WEIGHT;                                  // Add parent's weight
      if (C->NCHILD==0) {                                       // If leaf cell
        for (B_iter B=C->BODY; B!=C->BODY+C->NBODY; B++) {      //  Loop over bodies in cell
          B->WEIGHT += C->WEIGHT;                               //   Add cell weights to bodies
        }                                                       //  End loop over bodies in cell
      }                                                         // End if for leaf cell
#endif
#endif

      my_ityr::parallel_for<my_ityr::access_mode::read>(
          ityr::count_iterator<int>(0),
          ityr::count_iterator<int>(nchild),
          [=](int i) {
        preOrderTraversal(C0 + ichild + i, C0);
      });

      for (GC_iter CC=C0+ichild; CC!=C0+ichild+nchild; CC++) {// Loop over child cells
      }                                                         // End loop over chlid cells
    };

  public:
    //! Constructor
    UpDownPass(Kernel & _kernel) : kernel(_kernel) {}           // Initialize variables

    //! Upward pass (P2M, M2M)
    void upwardPass(GCells cells) {
      logger::startTimer("Upward pass");                        // Start timer
      my_ityr::root_spawn([=] {
        if (!cells.empty()) {                                     // If cell vector is not empty
          GC_iter C0 = cells.begin();                              //  Set iterator of target root cell
          my_ityr::parallel_for<my_ityr::access_mode::read_write>(
              cells.begin(), cells.end(), [=](Cell& c) {
            c.M.resize(kernel.NTERM, 0.0);                       //   Allocate & initialize M coefs
            c.L.resize(kernel.NTERM, 0.0);                       //   Allocate & initialize L coefs
          }, my_ityr::iro::block_size / sizeof(Cell));                                                       //  End loop over cells
          postOrderTraversal(C0, C0);                             //  Start post-order traversal from root
        }                                                         // End if for empty cell vector
      });
      logger::stopTimer("Upward pass");                         // Stop timer
    }

    //! Downward pass (L2L, L2P)
    void downwardPass(GCells cells) {
      logger::startTimer("Downward pass");                      // Start timer
      my_ityr::root_spawn([=] {
        if (!cells.empty()) {                                     // If cell vector is not empty
          GC_iter C0 = cells.begin();                              //  Root cell
          int ichild = C0->*(static_cast<int Cell::*>(&CellBase::ICHILD));
          int nchild = C0->*(static_cast<int Cell::*>(&CellBase::NCHILD));
          if (nchild == 0 ) {                                 //  If root is the only cell
            my_ityr::with_checkout_tied<my_ityr::access_mode::read>(
                C0, 1, [&](const Cell* C0_) {
              kernel.L2P(C0_);                                       //   L2P kernel
            });
          }                                                       //  End if root is the only cell
          for (GC_iter CC=C0+ichild; CC!=C0+ichild+nchild; CC++) {// Loop over child cells
            preOrderTraversal(CC, C0);                            //   Start pre-order traversal from root
          }                                                       //  End loop over child cells
        }                                                         // End if for empty cell vector
      });
      logger::stopTimer("Downward pass");                       // Stop timer
    }

    //! Get dipole of entire system
    vec3 getDipole(Bodies & bodies, vec3 X0) {
      vec3 dipole = 0;                                          // Initialize dipole correction
#if EXAFMM_LAPLACE
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {     // Loop over bodies
	dipole += (B->X - X0) * std::real(complex_t(B->SRC));   //  Calcuate dipole of the whole system
      }                                                         // End loop over bodies
#endif
      return dipole;                                            // Return dipole
    }

    //! Dipole correction
    void dipoleCorrection(Bodies & bodies, vec3 dipole, int numBodies, vec3 cycle) {
#if EXAFMM_LAPLACE
      real_t coef = 4 * M_PI / (3 * cycle[0] * cycle[1] * cycle[2]);// Precalcualte constant
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {     // Loop over bodies
	B->TRG[0] -= coef * norm(dipole) / numBodies / B->SRC;  //  Dipole correction for potential
	for (int d=0; d!=3; d++) {                              //  Loop over dimensions
	  B->TRG[d+1] -= coef * dipole[d];                      //   Dipole correction for forces
	}                                                       //  End loop over dimensions
      }                                                         // End loop over bodies
#endif
    }
  };
}
#endif
