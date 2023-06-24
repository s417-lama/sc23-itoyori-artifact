#include "args.h"
#include "bound_box.h"
#include "build_tree.h"
#include "dataset.h"
#include "kernel.h"
#include "logger.h"
#include "namespace.h"
#include "traversal.h"
#include "up_down_pass.h"
#include "verify.h"
using namespace EXAFMM_NAMESPACE;

void run_fmm(const Args& args) {
  const vec3 cycle = 2 * M_PI;
  const real_t eps2 = 0.0;
  const complex_t wavek = complex_t(10.,1.) / real_t(2 * M_PI);

  int my_rank = my_ityr::rank();
  int n_ranks = my_ityr::n_ranks();

  my_ityr::logger::init(my_rank, n_ranks);

  global_vec<Body> bodies_vec(global_vec_coll_opts);
  global_vec<Body> jbodies_vec(global_vec_coll_opts);
  global_vec<Body> buffer_vec(global_vec_coll_opts);
  global_vec<Cell> cells_vec(global_vec_coll_opts);

  GBodies bodies, jbodies, buffer;
  BoundBox boundBox;
  Bounds bounds;
  BuildTree buildTree(args.ncrit, args.nspawn);
  GCells cells, jcells;
  Dataset data;
  Kernel kernel(args.P, eps2, wavek);
  Traversal traversal(kernel, args.theta, args.nspawn, args.images, args.path);
  UpDownPass upDownPass(kernel);
  Verify verify(args.path);
  num_threads(args.threads);

  verify.verbose = args.verbose;
  logger::verbose = args.verbose;
  logger::path = args.path;

  bodies_vec.resize(args.numBodies);
  bodies = {bodies_vec.begin(), bodies_vec.end()};

  buffer_vec.resize(args.numBodies);
  buffer = {buffer_vec.begin(), buffer_vec.end()};

  if (my_rank == 0) {
    logger::printTitle("FMM Parameters");
    args.print(logger::stringLength);
  }

  if (args.IneJ) {
#if 0
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
      B->X[0] += M_PI;
      B->X[0] *= 0.5;
    }
    jbodies = data.initBodies(args.numBodies, args.distribution, 1);
    for (B_iter B=jbodies.begin(); B!=jbodies.end(); B++) {
      B->X[0] -= M_PI;
      B->X[0] *= 0.5;
    }
#endif
    std::cout << "IneJ unimplemented" << std::endl;
    abort();
  }

  bool pass = true;
  bool isTime = false;
  for (int t=0; t<args.repeat; t++) {
    my_ityr::master_do([=] {
      data.initBodies(bodies, args.distribution, 0);
    });

    if (my_rank == 0) {
      logger::printTitle("FMM Profiling");
      logger::startTimer("Total FMM");
      logger::startPAPI();
      logger::startDAG();
    }
    int numIteration = 1;
    if (isTime) numIteration = 10;
    for (int it=0; it<numIteration; it++) {
      if (my_rank == 0) {
        std::stringstream title;
        title << "Time average loop " << t;
        logger::printTitle(title.str());
      }

      bounds = boundBox.getBounds(bodies);

      if (args.IneJ) {
#if 0
        bounds = boundBox.getBounds(jbodies, bounds);
#endif
      }

      cells_vec = buildTree.buildTree(bodies, buffer, bounds);
      cells = {cells_vec.begin(), cells_vec.end()};

      if (my_rank == 0) {
        upDownPass.upwardPass(cells);
        traversal.initListCount(cells);
        traversal.initWeight(cells);
      }
      my_ityr::barrier();

      if (args.IneJ) {
#if 0
        jcells = buildTree.buildTree(jbodies, buffer, bounds);
        upDownPass.upwardPass(jcells);
        traversal.traverse(cells, jcells, cycle, args.dual);
#endif
      } else {
        traversal.traverse(cells, cells, cycle, args.dual);

        if (args.accuracy) {
          jbodies_vec = bodies_vec;
          jbodies = {jbodies_vec.begin(), jbodies_vec.end()};
        }
      }

      if (my_rank == 0) {
        upDownPass.downwardPass(cells);
      }
      my_ityr::barrier();
    }

    if (args.accuracy) {
      int should_break = my_ityr::master_do([&]() {
        logger::printTitle("Total runtime");
        logger::stopDAG();
        logger::stopPAPI();
        double totalFMM = logger::stopTimer("Total FMM");
        totalFMM /= numIteration;
        logger::resetTimer("Total FMM");
        if (args.write) {
          logger::writeTime();
        }
        traversal.writeList(cells, 0);

        if (!isTime) {
          const int numTargets = 100;

          global_vec<Body> bodies_sampled_vec = data.sampleBodies(bodies, numTargets);
          GBodies bodies_sampled = {bodies_sampled_vec.begin(), bodies_sampled_vec.end()};

          global_vec<Body> bodies2_vec = bodies_sampled_vec;
          GBodies bodies2 = {bodies2_vec.begin(), bodies2_vec.end()};

          data.initTarget(bodies_sampled);
          logger::startTimer("Total Direct");
          traversal.direct(bodies_sampled, jbodies, cycle);
          logger::stopTimer("Total Direct");

          ityr::with_checkout<my_ityr::access_mode::read_write, my_ityr::access_mode::read_write>(
              bodies_sampled, bodies2, [&](auto bodies_sampled_, auto bodies2_) {
            double potDif = verify.getDifScalar(bodies_sampled_, bodies2_);
            double potNrm = verify.getNrmScalar(bodies_sampled_);
            double accDif = verify.getDifVector(bodies_sampled_, bodies2_);
            double accNrm = verify.getNrmVector(bodies_sampled_);
            double potRel = std::sqrt(potDif/potNrm);
            double accRel = std::sqrt(accDif/accNrm);
            logger::printTitle("FMM vs. direct");
            verify.print("Rel. L2 Error (pot)",potRel);
            verify.print("Rel. L2 Error (acc)",accRel);

            buildTree.printTreeData(cells);
            traversal.printTraversalData();
            logger::printPAPI();

            pass = verify.regression(args.getKey(), isTime, t, potRel, accRel);
          });

          if (pass) {
            if (verify.verbose) std::cout << "passed accuracy regression at t: " << t << std::endl;
            /* if (args.accuracy) return 1; */
            /* t = -1; */
            /* isTime = true; */
            return 0;
          } else {
            if (verify.verbose) std::cout << "failed accuracy regression" << std::endl;
            return 1;
          }
        } else {
          /* pass = verify.regression(args.getKey(), isTime, t, totalFMM); */
          /* if (pass) { */
          /*   if (verify.verbose) std::cout << "passed time regression at t: " << t << std::endl; */
          /*   return 1; */
          /* } */
        }
        return 1;
      });

      if (should_break) break;
    } else {
      if (my_rank == 0) {
        buildTree.printTreeData(cells);
      }
    }
  }

  /* if (my_rank == 0) { */
  /*   if (!pass) { */
  /*     if (verify.verbose) { */
  /*       if(!isTime) std::cout << "failed accuracy regression" << std::endl; */
  /*       else std::cout << "failed time regression" << std::endl; */
  /*     } */
  /*     abort(); */
  /*   } */
  /*   if (args.getMatrix) { */
/* #if 0 */
  /*     traversal.writeMatrix(bodies, jbodies); */
/* #endif */
  /*   } */
  /*   logger::writeDAG(); */
  /* } */
  my_ityr::barrier();
}

int real_main(int argc, char ** argv) {
  Args args(argc, argv);
  my_ityr::iro::init(args.cache_size * 1024 * 1024, args.sub_block_size);

  run_fmm(args);

  my_ityr::iro::fini();
  return 0;
}

int main(int argc, char** argv) {
  my_ityr::main(real_main, argc, argv);
  return 0;
}
