#ifndef MADM_UTH_THREAD_H
#define MADM_UTH_THREAD_H

#include "../uth-cxx-decls.h"
#include "future.h"
#include "uni/context.h"

namespace madm {
namespace uth {

    template <class T, int NDEPS = 1>
    class thread {
        // shared object
        future<T, NDEPS> future_;

    public:
        // constr/destr with no thread
        thread();
        ~thread() = default;

        // constr create a thread
        template <class F, class... Args>
        explicit thread(const F& f, Args... args);

        // returns SYNCHED flag (false if stolen)
        template <class F, class... Args>
        bool spawn(const F& f, Args... args);

        template <class F, class ArgsTuple, class Callback>
        bool spawn_aux(const F& f, ArgsTuple args, Callback cb_on_die);

        T join(int dep_id = 0);

        template <class Callback>
        T join_aux(int dep_id, Callback cb_on_block);

        void discard(int dep_id = 0);

    private:
        template <class F, class ArgsTuple, class Callback>
        static void start(future<T, NDEPS> fut, F f, ArgsTuple args, Callback cb_on_die);
    };

    typedef madi::saved_context saved_context;

}
}

#endif
