#ifndef build_tree_serial_h
#define build_tree_serial_h
#include "logger.h"
#include "namespace.h"
#include "thread.h"
#include "types.h"

namespace EXAFMM_NAMESPACE {
  class BuildTree {
    // no parallelization is currently done here
    /* template <typename T> */
    /* using global_ptr = my_ityr::global_ptr<T>; */
    template <typename T>
    using global_ptr = T*;

    typedef vec<8,int> ivec8;                                   //!< Vector of 8 integer types

  private:
    //! Octree is used for building the FMM tree structure as "nodes", then transformed to "cells" data structure
    struct OctreeNode {
      int          IBODY;                                       //!< Index offset for first body in node
      int          NBODY;                                       //!< Number of descendant bodies
      int          NNODE;                                       //!< Number of descendant nodes
      global_ptr<OctreeNode> CHILD[8];                                    //!< Pointer to child node
      vec3         X;                                           //!< Coordinate at center
    };

    const int    ncrit;                                         //!< Number of bodies per leaf cell
    const int    nspawn;                                        //!< Threshold of NBODY for spawning new threads
    int          numLevels;                                     //!< Number of levels in tree
    GB_iter       B0;                                            //!< Iterator of first body
    global_ptr<OctreeNode> N0;                                            //!< Pointer to octree root node

  private:
    //! Counting bodies in each octant
    void countBodies(GBodies bodies, int begin, int end, vec3 X, ivec8 & NBODY) {
      for (int i=0; i<8; i++) NBODY[i] = 0;                     // Initialize number of bodies in octant
      my_ityr::serial_for<my_ityr::iro::access_mode::read,
                          my_ityr::iro::access_mode::read>(
          ityr::count_iterator<int>(begin),
          ityr::count_iterator<int>(end),
          bodies.begin() + begin,
          [&](int i, const auto& B) {

        vec3 x = B.X;                                   //  Coordinates of body
        if (B.ICELL < 0) {                                //  If using residual index
          auto mp_X = static_cast<vec3 Body::*>(&Source::X);
          x = (&bodies[i+B.ICELL])->*(mp_X);                      //   Use coordinates of first body in residual group
        }
        int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);// Which octant body belongs to
        NBODY[octant]++;                                        //  Increment body count in octant
                                                                //
      }, my_ityr::iro::block_size);
    }

    //! Sorting bodies according to octant (Morton order)
    void moveBodies(GBodies bodies, GBodies buffer, int begin, int end,
                           ivec8 octantOffset, vec3 X) {
      my_ityr::serial_for<my_ityr::iro::access_mode::read,
                          my_ityr::iro::access_mode::read>(
          ityr::count_iterator<int>(begin),
          ityr::count_iterator<int>(end),
          bodies.begin() + begin,
          [&](int i, const auto& B) {

        vec3 x = B.X;                                   //  Coordinates of body
        if (B.ICELL < 0) {                                //  If using residual index
          auto mp_X = static_cast<vec3 Body::*>(&Source::X);
          x = (&bodies[i+B.ICELL])->*(mp_X);                      //   Use coordinates of first body in residual group
        }
        int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);// Which octant body belongs to`
        buffer[octantOffset[octant]] = B;               //   Permute bodies out-of-place according to octant
        octantOffset[octant]++;                                 //  Increment body count in octant
                                                                //
      }, my_ityr::iro::block_size);
    }

    global_ptr<OctreeNode> alloc_node() const {
      /* return my_ityr::iro::malloc_local<OctreeNode>(1); */
      return reinterpret_cast<global_ptr<OctreeNode>>(std::malloc(sizeof(OctreeNode)));
    }

    void free_node(global_ptr<OctreeNode> node) const {
      /* my_ityr::iro::free(node, 1); */
      std::free(node);
    }

    //! Create an octree node
    global_ptr<OctreeNode> makeOctNode(int begin, int end, vec3 X, bool nochild) {
      global_ptr<OctreeNode> octNode_ = alloc_node();
      /* global_ptr<OctreeNode> octNode = alloc_node(); */
      /* my_ityr::with_checkout_tied<my_ityr::access_mode::write>( */
      /*     octNode, 1, [&](OctreeNode* octNode_) { */
        OctreeNode* octNode = new (octNode_) OctreeNode();
        octNode->IBODY = begin;                                   // Index of first body in node
        octNode->NBODY = end - begin;                             // Number of bodies in node
        octNode->NNODE = 1;                                       // Initialize counter for decendant nodes
        octNode->X = X;                                           // Center coordinates of node
        if (nochild) {                                            // If node has no children
          for (int i=0; i<8; i++) octNode->CHILD[i] = nullptr;       //  Initialize pointers to children
        }                                                         // End if for node children
      /* }); */
      return octNode;                                           // Return node
    }

    //! Exclusive scan with offset
    inline ivec8 exclusiveScan(ivec8 input, int offset) {
      ivec8 output;                                             // Output vector
      for (int i=0; i<8; i++) {                                 // Loop over elements
        output[i] = offset;                                     //  Set value
        offset += input[i];                                     //  Increment offset
      }                                                         // End loop over elements
      return output;                                            // Return output vector
    }

    //! Recursive functor for building nodes of an octree adaptively using a top-down approach
    global_ptr<OctreeNode> buildNodes(GBodies bodies, GBodies buffer,
                           int begin, int end, vec3 X, real_t R0,
                           int level=0, bool direction=false) {
      if (begin == end) {                                       // If no bodies are left
        return nullptr;                                                 //  End buildNodes()
      }                                                         // End if for no bodies
      if (end - begin <= ncrit) {                               // If number of bodies is less than threshold
        if (direction) {                                          //  If direction of data is from bodies to buffer
          my_ityr::with_checkout_tied<my_ityr::access_mode::read,
                                      my_ityr::access_mode::write>(
              bodies.begin() + begin, end - begin,
              buffer.begin() + begin, end - begin,
              [&](const Body* b_src, Body* b_dest) {
            for (int i = 0; i < end - begin; i++) {
              b_dest[i] = b_src[i];
            }
          });
          /* my_ityr::parallel_transform(bodies.begin() + begin, */
          /*                             bodies.begin() + end, */
          /*                             buffer.begin() + begin, */
          /*                             [](const auto& B) { return B; }, */
          /*                             my_ityr::iro::block_size); */
        }
        return makeOctNode(begin,end,X,true);        //  Create an octree node and assign it's pointer
      }                                                         // End if for number of bodies
      global_ptr<OctreeNode> octNode = makeOctNode(begin,end,X,false);         // Create an octree node with child nodes
      ivec8 NBODY;                                              // Number of bodies in node
      countBodies(bodies, begin, end, X, NBODY);                // Count bodies in each octant
      ivec8 octantOffset = exclusiveScan(NBODY, begin);         // Exclusive scan to obtain offset from octant count
      moveBodies(bodies, buffer, begin, end, octantOffset, X);  // Sort bodies according to octant
                                                                //
      auto children = global_ptr<global_ptr<OctreeNode>>(&(octNode->*(&OctreeNode::CHILD)));
      for (int i=0; i<8; i++) {                                 // Loop over children
        vec3 Xchild = X;                                        //  Initialize center coordinates of child node
        real_t r = R0 / (1 << (level + 1));                     //  Radius of cells for child's level
        for (int d=0; d<3; d++) {                               //  Loop over dimensions
          Xchild[d] += r * (((i & 1 << d) >> d) * 2 - 1);       //   Shift center coordinates to that of child node
        }                                                       //  End loop over dimensions
        children[i] = buildNodes(buffer, bodies,           //  Recursive call for children
                   octantOffset[i], octantOffset[i] + NBODY[i],
                   Xchild, R0, level+1, !direction);
      }                                                         // End loop over children
      for (int i=0; i<8; i++) {                                 // Loop over children
        if (global_ptr<OctreeNode>(children[i]))
          octNode->*(&OctreeNode::NNODE) += global_ptr<OctreeNode>(children[i])->*(&OctreeNode::NNODE);// If child exists increment child node count
      }                                                         // End loop over chlidren
      return octNode;
    }

    //! Get Morton key
    uint64_t getKey(vec3 X, vec3 Xmin, real_t diameter, int level) const {
      int iX[3] = {0, 0, 0};                                    // Initialize 3-D index
      for (int d=0; d<3; d++) iX[d] = int((X[d] - Xmin[d]) / diameter);// 3-D index
      uint64_t index = ((1 << 3 * level) - 1) / 7;              // Levelwise offset
      for (int l=0; l<level; l++) {                             // Loop over levels
        for (int d=0; d<3; d++) index += (iX[d] & 1) << (3 * l + d); // Interleave bits into Morton key
        for (int d=0; d<3; d++) iX[d] >>= 1;                    //  Bitshift 3-D index
      }                                                         // End loop over levels
      return index;                                             // Return Morton key
    }

    //! Creating cell data structure from nodes
    void nodes2cells(global_ptr<OctreeNode> octNode, GC_iter C,
                     GC_iter C0, GC_iter CN, vec3 X0, real_t R0,
                     int & maxLevel, int level=0, int iparent=0) {
      const OctreeNode* o = octNode;
      my_ityr::with_checkout<my_ityr::access_mode::write>(
          C, 1,
          [&](Cell* c_) {
      /* my_ityr::with_checkout<my_ityr::access_mode::read, */
      /*                        my_ityr::access_mode::write>( */
      /*     octNode, 1, C, 1, */
      /*     [&](const OctreeNode* o, Cell* c_) { */
        Cell* c = new (c_) Cell();
        c->IPARENT = iparent;                                     //  Index of parent cell
        c->R       = R0 / (1 << level);                           //  Cell radius
        c->X       = o->X;                                  //  Cell center
        c->NBODY   = o->NBODY;                              //  Number of decendant bodies
        c->IBODY   = o->IBODY;                              //  Index of first body in cell
        c->BODY    = B0 + c->IBODY;                               //  Iterator of first body in cell
        c->ICELL   = getKey(c->X, X0-R0, 2*c->R, level);          //  Get Morton key
        if (o->NNODE == 1) {                                //  If node has no children
          c->ICHILD = 0;                                          //   Set index of first child cell to zero
          c->NCHILD = 0;                                          //   Number of child cells
          assert(c->NBODY > 0);                                   //   Check for empty leaf cells
          maxLevel = std::max(maxLevel, level);                   //   Update maximum level of tree
        } else {                                                  //  Else if node has children
          int nchild = 0;                                         //   Initialize number of child cells
          int octants[8];                                         //   Map of child index to octants
          for (int i=0; i<8; i++) {                               //   Loop over octants
            if (o->CHILD[i]) {                              //    If child exists for that octant
              octants[nchild] = i;                                //     Map octant to child index
              nchild++;                                           //     Increment child cell counter
            }                                                     //    End if for child existance
          }                                                       //   End loop over octants
          GC_iter Ci = CN;                                         //   CN points to the next free memory address
          c->ICHILD = Ci - C0;                                    //   Set Index of first child cell
          c->NCHILD = nchild;                                     //   Number of child cells
          assert(C->NCHILD > 0);                                  //   Check for childless non-leaf cells
          CN += nchild;                                           //   Increment next free memory address
          for (int i=0; i<nchild; i++) {                          //   Loop over children
            int octant = octants[i];                              //    Get octant from child index
            nodes2cells(o->CHILD[octant], Ci, C0, CN,       //    Recursive call for child cells
                        X0, R0, numLevels, level+1, C-C0);
            Ci++;                                                 //    Increment cell iterator
            CN += o->CHILD[octant]->*(&OctreeNode::NNODE) - 1;              //    Increment next free memory address
          }                                                       //   End loop over children
          for (int i=0; i<nchild; i++) {                          //   Loop over children
            int octant = octants[i];                              //    Get octant from child index
            free_node(o->CHILD[octant]);
          }                                                       //   End loop over children
          maxLevel = std::max(maxLevel, level+1);                 //   Update maximum level of tree
        }                                                         //  End if for child existance
      });
    };

    //! Transform Xmin & Xmax to X (center) & R (radius)
    Box bounds2box(Bounds bounds) {
      vec3 Xmin = bounds.Xmin;                                  // Set local Xmin
      vec3 Xmax = bounds.Xmax;                                  // Set local Xmax
      Box box;                                                  // Bounding box
      for (int d=0; d<3; d++) box.X[d] = (Xmax[d] + Xmin[d]) / 2; // Calculate center of domain
      box.R = 0;                                                // Initialize localRadius
      for (int d=0; d<3; d++) {                                 // Loop over dimensions
	box.R = std::max(box.X[d] - Xmin[d], box.R);            //  Calculate min distance from center
	box.R = std::max(Xmax[d] - box.X[d], box.R);            //  Calculate max distance from center
      }                                                         // End loop over dimensions
      box.R *= 1.00001;                                         // Add some leeway to radius
      return box;                                               // Return box.X and box.R
    }

  public:
    BuildTree(int _ncrit, int _nspawn) : ncrit(_ncrit), nspawn(_nspawn), numLevels(0) {}

    //! Build tree structure top down
    global_vec<Cell> buildTree(GBodies bodies, GBodies buffer, Bounds bounds) {
      int my_rank = my_ityr::rank();

      if (my_rank == 0) {
        logger::startTimer("Grow tree");                          // Start timer
      }

      Box box = bounds2box(bounds);                             // Get box from bounds
      if (bodies.empty()) {                                     // If bodies vector is empty
        N0 = nullptr;                                              //  Reinitialize N0 with NULL
      } else {                                                  // If bodies vector is not empty
#if 0
        if (bodies.size() > buffer.size()) buffer.resize(bodies.size());// Enlarge buffer if necessary
#else
        assert(bodies.size() <= buffer.size());
#endif
        assert(box.R > 0);                                      // Check for bounds validity
        B0 = bodies.begin();                                    // Bodies iterator

        my_ityr::barrier();

        N0 = my_ityr::master_do([=]() {
          return buildNodes(bodies, buffer, 0, bodies.size(),        // Build octree nodes
                            box.X, box.R);
        });
      }

      if (my_rank == 0) {
        logger::stopTimer("Grow tree");                           // Stop timer
        logger::startTimer("Link tree");                          // Start timer
      }

      global_vec<Cell> cells_vec(global_vec_coll_opts);                                              // Initialize cell array

      if (N0 != nullptr) {                                         // If the node tree is not empty
        /* std::size_t ncells = N0->*(&OctreeNode::NNODE); */
        std::size_t ncells = my_ityr::master_do([=]() { return N0->*(&OctreeNode::NNODE); });

        /* if (my_rank == 0) printf("ncells: %ld\n", ncells); */

        cells_vec.resize(ncells);
        if (my_rank == 0) {
          GC_iter C0 = cells_vec.begin();                              //  Cell begin iterator
          nodes2cells(N0, C0, C0, C0+1, box.X, box.R, numLevels); // Instantiate recursive functor
          free_node(N0);
        }
      }                                                         // End if for empty node tree
      my_ityr::barrier();

      if (my_rank == 0) {
        logger::stopTimer("Link tree");                           // Stop timer
      }
      return cells_vec;                                             // Return cells array
    }

    //! Print tree structure statistics
    void printTreeData(GCells cells) {
      if (logger::verbose && !cells.empty()) {                  // If verbose flag is true
	logger::printTitle("Tree stats");                       //  Print title
        int nbody = cells.begin()->*(static_cast<int Cell::*>(&CellBase::NBODY));
	std::cout  << std::setw(logger::stringLength) << std::left//  Set format
		   << "Bodies"     << " : " << nbody << std::endl// Print number of bodies
		   << std::setw(logger::stringLength) << std::left//  Set format
		   << "Cells"      << " : " << cells.size() << std::endl// Print number of cells
		   << std::setw(logger::stringLength) << std::left//  Set format
		   << "Tree depth" << " : " << numLevels << std::endl;//  Print number of levels
      }                                                         // End if for verbose flag
    }
  };
}
#endif
