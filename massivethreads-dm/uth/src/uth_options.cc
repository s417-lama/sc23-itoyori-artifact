#include "uth_options.h"
#include <cstdlib>
#include <cstdio>
#include <unistd.h>

#include <mpi.h>

namespace madi {

    // default values of configuration variables
    struct uth_options uth_options = {
        1024 * 1024,        // stack_size
        1,                  // stack_overflow_detection
        1024,               // taskq_capacity
        8192,               // page_size
        0,                  // profile
        0,                  // steal_log
        1,                  // aborting_steal
    };

    template <class T>
    void set_option(const char *name, T *value);

    template <>
    void set_option<int>(const char *name, int *value) {
        char *s = getenv(name);
        if (s != NULL)
            *value = atoi(s);
    }

    template <>
    void set_option<size_t>(const char *name, size_t *value) {
        char *s = getenv(name);

        if (s != NULL) {
            *value = static_cast<size_t>(atol(s));
        }
    }

    template <class T>
    void set_option_coll(const char *name, T *value) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0) {
            set_option(name, value);
        }

        MPI_Bcast(value, sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    void uth_options_initialize()
    {
        set_option_coll("MADM_STACK_SIZE", &uth_options.stack_size);
        set_option_coll("MADM_STACK_DETECT", &uth_options.stack_overflow_detection);
        set_option_coll("MADM_TASKQ_CAPACITY", &uth_options.taskq_capacity);
        set_option_coll("MADM_PROFILE", &uth_options.profile);
        set_option_coll("MADM_STEAL_LOG", &uth_options.steal_log);
        set_option_coll("MADM_ABORTING_STEAL", &uth_options.aborting_steal);

        long page_size = sysconf(_SC_PAGE_SIZE);
        uth_options.page_size = static_cast<size_t>(page_size);
    }

    void uth_options_finalize()
    {
    }

    void uth_options_print(FILE *f)
    {
        fprintf(f,
                "MADM_STACK_SIZE = %zu"
                ", MADM_TASKQ_CAPACITY = %zu"
                ", MADM_PROFILE = %d"
                ", MADM_STEAL_LOG = %d"
                ", MADM_ABORTING_STEAL = %d"
                "\n",
                uth_options.stack_size,
                uth_options.taskq_capacity,
                uth_options.profile,
                uth_options.steal_log,
                uth_options.aborting_steal);
    }
}
