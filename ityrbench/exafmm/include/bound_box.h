#ifndef bound_box_h
#define bound_box_h
#include "logger.h"
#include "namespace.h"
#include "types.h"

namespace EXAFMM_NAMESPACE {
  class BoundBox {
  public:
    //! Get Xmin and Xmax of bodies
    Bounds getBounds(Bodies & bodies) {
      logger::startTimer("Get bounds");                         // Start timer
      Bounds bounds;                                            // Bounds : Contains Xmin, Xmax
      if (bodies.empty()) {                                     // If body vector is empty
	bounds.Xmin = bounds.Xmax = 0;                          //  Set bounds to 0
      } else {                                                  // If body vector is not empty
	bounds.Xmin = bounds.Xmax = bodies.front().X;           //  Initialize Xmin, Xmax
        for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {   //  Loop over bodies
          bounds.Xmin = min(B->X, bounds.Xmin - 1e-5);          //   Update Xmin
          bounds.Xmax = max(B->X, bounds.Xmax + 1e-5);          //   Update Xmax
        }                                                       //  End loop over bodies
      }                                                         // End if for empty body vector
      logger::stopTimer("Get bounds");                          // Stop timer
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Update Xmin and Xmax of bodies
    Bounds getBounds(Bodies bodies, Bounds bounds) {
      logger::startTimer("Get bounds");                         // Start timer
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {     // Loop over bodies
        bounds.Xmin = min(B->X, bounds.Xmin - 1e-5);            //  Update Xmin
        bounds.Xmax = max(B->X, bounds.Xmax + 1e-5);            //  Update Xmax
      }                                                         // End loop over bodies
      logger::stopTimer("Get bounds");                          // Stop timer
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Get Xmin and Xmax of cells
    Bounds getBounds(Cells cells) {
      logger::startTimer("Get bounds");                         // Start timer
      Bounds bounds;                                            // Bounds : Contains Xmin, Xmax
      if (cells.empty()) {                                      // If cell vector is empty
	bounds.Xmin = bounds.Xmax = 0;                          //  Set bounds to 0
      } else {                                                  // If cell vector is not empty
	bounds.Xmin = bounds.Xmax = cells.front().X;            //  Initialize Xmin, Xmax
        for (C_iter C=cells.begin(); C!=cells.end(); C++) {     //  Loop over cells
          bounds.Xmin = min(C->X - 1e-5, bounds.Xmin);          //   Update Xmin
          bounds.Xmax = max(C->X + 1e-5, bounds.Xmax);          //   Update Xmax
        }                                                       //  End loop over cells
      }                                                         // End if for empty body vector
      logger::stopTimer("Get bounds");                          // Stop timer
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Update Xmin and Xmax of cells
    Bounds getBounds(Cells cells, Bounds bounds) {
      logger::startTimer("Get bounds");                         // Start timer
      for (C_iter C=cells.begin(); C!=cells.end(); C++) {       // Loop over cells
        bounds.Xmin = min(C->X - 1e-5, bounds.Xmin);            //  Update Xmin
        bounds.Xmax = max(C->X + 1e-5, bounds.Xmax);            //  Update Xmax
      }                                                         // End loop over cells
      logger::stopTimer("Get bounds");                          // Stop timer
      return bounds;                                            // Return Xmin and Xmax
    }

    // Global
    // -----------------------------------------
    // TODO: coarse-grained checkin/checkout
    // What are the stuffs like 1e-5...? We cannot parallelize...

    //! Get Xmin and Xmax of bodies
    Bounds getBounds(GBodies bodies) {
      int my_rank = my_ityr::rank();
      if (my_rank == 0) {
        logger::startTimer("Get bounds");                         // Start timer
      }
      Bounds bounds;                                            // Bounds : Contains Xmin, Xmax
      if (bodies.empty()) {                                     // If body vector is empty
	bounds.Xmin = bounds.Xmax = 0;                          //  Set bounds to 0
      } else {                                                  // If body vector is not empty
        auto mp_X = static_cast<vec3 Body::*>(&Source::X);
        bounds.Xmin = bounds.Xmax = bodies.begin()->*(mp_X);
        my_ityr::serial_for<my_ityr::iro::access_mode::read>(
            bodies.begin(),
            bodies.end(),
            [&](const auto& B) {
          bounds.Xmin = min(B.X, bounds.Xmin - 1e-5);          //   Update Xmin
          bounds.Xmax = max(B.X, bounds.Xmax + 1e-5);          //   Update Xmax
        }, my_ityr::iro::block_size);
      }                                                         // End if for empty body vector
      if (my_rank == 0) {
        logger::stopTimer("Get bounds");                          // Stop timer
      }
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Update Xmin and Xmax of bodies
    Bounds getBounds(GBodies bodies, Bounds bounds) {
      int my_rank = my_ityr::rank();
      if (my_rank == 0) {
        logger::startTimer("Get bounds");                         // Start timer
      }
      my_ityr::serial_for<my_ityr::iro::access_mode::read>(
          bodies.begin(),
          bodies.end(),
          [&](const auto& B) {
        bounds.Xmin = min(B.X, bounds.Xmin - 1e-5);            //  Update Xmin
        bounds.Xmax = max(B.X, bounds.Xmax + 1e-5);            //  Update Xmax
      }, my_ityr::iro::block_size);
      if (my_rank == 0) {
        logger::stopTimer("Get bounds");                          // Stop timer
      }
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Get Xmin and Xmax of cells
    Bounds getBounds(GCells cells) {
      int my_rank = my_ityr::rank();
      if (my_rank == 0) {
        logger::startTimer("Get bounds");                         // Start timer
      }
      Bounds bounds;                                            // Bounds : Contains Xmin, Xmax
      if (cells.empty()) {                                      // If cell vector is empty
	bounds.Xmin = bounds.Xmax = 0;                          //  Set bounds to 0
      } else {                                                  // If cell vector is not empty
        auto mp_X = static_cast<vec3 Cell::*>(&CellBase::X);
	bounds.Xmin = bounds.Xmax = cells.begin()->*(mp_X);           //  Initialize Xmin, Xmax
        my_ityr::serial_for<my_ityr::iro::access_mode::read>(
            cells.begin(),
            cells.end(),
            [&](const auto& C) {
          bounds.Xmin = min(vec3(C.X) - 1e-5, bounds.Xmin);          //   Update Xmin
          bounds.Xmax = max(vec3(C.X) + 1e-5, bounds.Xmax);          //   Update Xmax
        }, my_ityr::iro::block_size);
      }                                                         // End if for empty body vector
      if (my_rank == 0) {
        logger::stopTimer("Get bounds");                          // Stop timer
      }
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Update Xmin and Xmax of cells
    Bounds getBounds(GCells cells, Bounds bounds) {
      int my_rank = my_ityr::rank();
      if (my_rank == 0) {
        logger::startTimer("Get bounds");                         // Start timer
      }
      my_ityr::serial_for<my_ityr::iro::access_mode::read>(
          cells.begin(),
          cells.end(),
          [&](const auto& C) {
        bounds.Xmin = min(vec3(C.X) - 1e-5, bounds.Xmin);            //  Update Xmin
        bounds.Xmax = max(vec3(C.X) + 1e-5, bounds.Xmax);            //  Update Xmax
      }, my_ityr::iro::block_size);
      if (my_rank == 0) {
        logger::stopTimer("Get bounds");                          // Stop timer
      }
      return bounds;                                            // Return Xmin and Xmax
    }
  };
}
#endif
