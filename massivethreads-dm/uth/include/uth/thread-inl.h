#ifndef MADM_UTH_THREAD_INL_H
#define MADM_UTH_THREAD_INL_H

#include "future.h"
#include "future-inl.h"
#include "uni/worker-inl.h"
#include <tuple>

namespace madm {
namespace uth {

    template <class T, int NDEPS>
    thread<T, NDEPS>::thread() : future_() {}

    template <class T, int NDEPS>
    template <class F, class... Args>
    thread<T, NDEPS>::thread(const F& f, Args... args)
        : future_()
    {
        spawn(f, args...);
    }

    template <class T, int NDEPS>
    template <class F, class... Args>
    bool thread<T, NDEPS>::spawn(const F& f, Args... args)
    {
        return spawn_aux(f, std::make_tuple(args...), [](bool parent_popped){});
    }

    template <class T, int NDEPS>
    template <class F, class ArgsTuple, class Callback>
    bool thread<T, NDEPS>::spawn_aux(const F& f, ArgsTuple args, Callback cb_on_die)
    {
        madi::logger::checkpoint<madi::logger::kind::WORKER_BUSY>();

        madi::worker& w = madi::current_worker();
        future_ = future<T, NDEPS>::make(w);

        return w.fork(start<F, ArgsTuple, Callback>, std::make_tuple(future_, f, args, cb_on_die));
    }

    template <class T, int NDEPS>
    T thread<T, NDEPS>::join(int dep_id)
    {
        return join_aux(dep_id, []{});
    }

    template <class T, int NDEPS>
    template <class Callback>
    T thread<T, NDEPS>::join_aux(int dep_id, Callback cb_on_block)
    {
        T ret = future_.get(dep_id, cb_on_block);
        return ret;
    }

    template <class T, int NDEPS>
    void thread<T, NDEPS>::discard(int dep_id)
    {
        return future_.discard(dep_id);
    }

    template <class T, int NDEPS>
    template <class F, class ArgsTuple, class Callback>
    void thread<T, NDEPS>::start(future<T, NDEPS> fut, F f, ArgsTuple args, Callback cb_on_die)
    {
        madi::logger::checkpoint<madi::logger::kind::WORKER_THREAD_FORK>();

        T value = std::apply(f, args);

        madi::logger::checkpoint<madi::logger::kind::WORKER_BUSY>();

        fut.set(value, cb_on_die);
    }

    template <int NDEPS>
    class thread<void, NDEPS> {
    private:
        future<long, NDEPS> future_;

    public:
        // constr/destr with no thread
        thread()  = default;
        ~thread() = default;

        // constr create a thread
        template <class F, class... Args>
        explicit thread(const F& f, Args... args)
            : future_()
        {
            spawn(f, args...);
        }

        template <class F, class... Args>
        bool spawn(const F& f, Args... args)
        {
            return spawn_aux(f, std::make_tuple(args...), [](bool parent_popped){});
        }

        template <class F, class ArgsTuple, class Callback>
        bool spawn_aux(const F& f, ArgsTuple args, Callback cb_on_die)
        {
            madi::logger::checkpoint<madi::logger::kind::WORKER_BUSY>();

            madi::worker& w = madi::current_worker();
            future_ = future<long, NDEPS>::make(w);

            return w.fork(start<F, ArgsTuple, Callback>, std::make_tuple(future_, f, args, cb_on_die));
        }

        // copy and move constrs
        thread& operator=(const thread&) = delete;
        thread(thread&& other);  // TODO: implement

        void join(int dep_id = 0) { join_aux(dep_id, []{}); }
        template <class Callback>
        void join_aux(int dep_id, Callback cb_on_block) { future_.get(dep_id, cb_on_block); }
        void discard(int dep_id) { return future_.discard(dep_id); }

    private:
        template <class F, class ArgsTuple, class Callback>
        static void start(future<long, NDEPS> fut, F f, ArgsTuple args, Callback cb_on_die)
        {
            madi::logger::checkpoint<madi::logger::kind::WORKER_THREAD_FORK>();

            std::apply(f, args);

            madi::logger::checkpoint<madi::logger::kind::WORKER_BUSY>();

            long value = 0;
            fut.set(value, cb_on_die);
        }
    };

}
}

#endif
