# This file is part of pyrerp
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# support multiple iterables like imap (with same semantics: stop when first
# one is exhausted)

# proper design:
#   object representing a worker, gets started and then jobs are queued to it
#     directly. jobs come out into a single queue.
#   (if too many finished jobs pile up, need backpressure -- keep track of
#   which workers are idle)

# if debugging and continue-on-error enabled, put the erroring worker to one
# side and spawn a new one.
# make debug-start an inline command
# and have debug-input and debug-output as commands?
# or if we have a tty to offer, debug-tty to run on it directly
# handling qtconsole + windows + cross-machine debuggin is going to be a
# hassle. spawning an ipython kernel may help?

# unordered parallel imap
# basically the same as multiprocessing.Pool.imap_unordered
# but with some pain points removed.
#
# serial in-process/parallel out-of-process/number of processes are controlled
# by an envvar or global setting, so no need to modify all your actual
# computational loops every time you need to debug.
#
# control-C handling:
# - if worker processes see a control-C, they just silently quit
# - if parent object is destructed, silently kill all children
# - if a worker process disappears, parent just raises an error on next call
#   to .next()
#   (fancier option, not implemented: respawn the child and resubmit the work)
#
# debugging:
# some questionable shenanigans are used to
# when a worker raises an error, it's re-raised by the parent, fine. But the
# worker keeps the stack frame around, and the parent remembers which worker
# it is, so that it can run a command which:
#   - wakes up a pdb post-mortem in the worker
#   - blocks the parent
#   - unblocks the parent when the post-mortem finishes
# for now, we can just make failures final... like 'map' does. continuing
# after a failure is not really critical for lots of cases.

# other useful stuff that might be useful to add in the future:
# continuing after one job raises an exception?
# if a child dies, automatic respawn and resubmit of affected jobs?
# tricks for better IPC: picloud-style pickling, fork sharing?
# support for other parallelization systems (ssh, slurm, sge, picloud)?
# chunking?
# passing numpy arrays through shared memory? (not clear if this is really
#   even a win)
# coordination between multiple parimaps, allowing you to have multiple
#   instantiated at once (e.g., one feeding another), and coordinate the work
#   using a shared process pool? (Deadlock avoidance becomes a bit tricky!)

import os
import multiprocessing
import itertools
import traceback
import sys

CONFIG_ENVVAR = "PYRERP_PARALLEL"

class ParimapError(Exception):
    pass

def _try_switch_class(obj, new_class):
    # If this works, then definitely easiest
    try:
        obj.__class__ = new_class
    except TypeError:
        pass
    else:
        return obj
    # Don't even try to handle old-style classes
    if not isinstance(obj, object):
        return obj
    # new-style classes always have a __reduce_ex__ method
    reduced = obj.__reduce_ex__(2)
    if not isinstance(reduced, tuple) or not 2 <= len(reduced) <= 5:
        # broken reduce method
        return obj
    reduced += (None,) * (5 - len(reduced))
    constructor, constructor_args, state, seq, mapping = reduced
    if constructor is not obj.__class__:
        # Something clever going on, give up
        return obj
    new_obj = new_class(*constructor_args)
    if state is not None:
        if hasattr(new_obj, "__setstate__"):
            new_obj.__setstate__(state)
        else:
            new_obj.__dict__.update(state)
    if seq is not None:
        new_obj.extend(seq)
    if mapping is not None:
        for key, value in mapping:
            new_obj[key] = value
    return new_obj

class _CrossProcessExceptionMixin(object):
    # assumes attrs:
    #   -- _parimap_orig_class
    #   -- _parimap_traceback_list
    #   -- _parimap_mapper
    def pm(self):
        return self._parimap_mapper.post_mortem()

    def __str__(self):
        # nose likes to perform some shenanigans, where when it catches one
        # exception it creates a new one by calling
        #   exc.__type__(modified_exc_string)
        # this creates an object that will normally print the same as the
        # original exception, but is missing our special stuff.
        if not hasattr(self, "_parimap_traceback_list"):
            return super(_CrossProcessExceptionMixin, self).__str__()
        try:
            return ("\n-- Traceback from worker process --\n%s"
                    "%s: %s\n\n"
                    "If running from interactive shell, to open a debugger use:\n"
                    "   import sys; sys.last_value.pm()"
                    % ("".join(traceback.format_list(self._parimap_traceback_list)),
                       self._parimap_orig_class.__name__,
                       self._parimap_orig_class.__str__(self),
                       ))
        except Exception, e:
            return "<unprintable %s b/c of %s>" % (
                self.__class__.__name__, e)

def _try_wrap_exception(exc, traceback_list, mapper):
    if not isinstance(exc, object):
        return exc
    old_type = exc.__class__
    new_type = type.__new__(type,
                            # Class name
                            "CrossProcess__" + old_type.__name__,
                            # Bases
                            (_CrossProcessExceptionMixin, old_type),
                            # __dict__
                            {})
    new_exc = _try_switch_class(exc, new_type)
    if isinstance(new_exc, new_type):
        new_exc._parimap_orig_class = old_type
        new_exc._parimap_traceback_list = traceback_list
        new_exc._parimap_mapper = mapper
    return new_exc

_config = {
    "mode": "multiprocess",
    "processes": "auto",
    }

if CONFIG_ENVVAR in os.environ:
    for tag in os.environ["CONFIG_ENVVAR"].split(","):
        if tag in ("serial", "parallel"):
            _config["mode"] = tag
            continue
        if "=" in tag:
            key, value = tag.split("=", 1)
            if key == "processes":
                _config[key] = int(value)
                continue
        raise ValueError("envvar $%s contains ill-formed config tag %r"
                         % (CONFIG_ENVVAR, tag))

def configure(mode=None, processes=None):
    if mode is not None:
        self._config["mode"] = mode
    if processes is not None:
        self._config["processes"] = processes

class _OrderPreserver(object):
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return (args[0], self._fn(*args[1:], **kwargs))

def parimap(fn, *iterables, **kwargs):
    if _config["mode"] == "serial":
        for result in parimap_unordered(fn, *iterables, **kwargs):
            yield result
    else:
        iterables = (itertools.count(),) + iterables
        result_idxs = {}
        next_result_idx = 0
        for idx, result in parimap_unordered(_OrderPreserver(fn),
                                             *iterables, **kwargs):
            result_idxs[idx] = result
            while next_result_idx in result_idxs:
                result = result_idxs[next_result_idx]
                del result_idxs[next_result_idx]
                yield result
                next_result_idx += 1


def parimap_unordered(fn, *iterables, **kwargs):
    iterators = [iter(obj) for obj in iterables]
    special_args = {}
    for key in kwargs:
        if key.startswith("__"):
            special_args[key[2:]] = kwargs[key]
            del kwargs[key]
    if _config["mode"] == "serial":
        return itertools.imap(fn, *iterators, **kwargs)
    elif _config["mode"] == "multiprocess":
        return MPimap(fn, iterators, kwargs, special_args)

def _mpimap_worker(worker_id, fn, kwargs,
                   work_queue, result_queue,
                   debug_start_queue, debug_done_queue):
    try:
        get = work_queue.get
        put = result_queue.put
        while True:
            iterator_values = get()
            try:
                result = fn(*iterator_values, **kwargs)
            except BaseException, e:
                _, _, tb = sys.exc_info()
                put(("exc", e, traceback.extract_tb(tb), worker_id))
                # Block here and wait in case the user wants to debug
                while True:
                    debug_start_queue.get()
                    try:
                        print "PDB in worker: starting"
                        # One of the first things multiprocessing executes in
                        # a child process is:
                        #   sys.stdin.close()
                        #   sys.stdin = open(os.devnull)
                        # Of course this totally messes up PDB.
                        #
                        # For some reason I don't understand, on Linux fd 0
                        # seems to remain open and available, so we can just
                        # reassign sys.stdin. Some other hack may be required
                        # on Windows.
                        sys.stdin = os.fdopen(0)
                        import pdb
                        value = pdb.post_mortem()
                    except Exception, e:
                        print "PDB in worker: raised error, quitting"
                        debug_done_queue.put(("debug-exc", e))
                    else:
                        print "PDB in worker: finished, returning to parent"
                        debug_done_queue.put(("debug-success", value))
            else:
                put(("success", result))
    except KeyboardInterrupt:
        # silently exit
        pass

class MPimap(object):
    _RUNNING = object()
    _DEBUGGABLE = object()
    _CRASHED = object()
    _CLOSED = object()
    _FINISHED = object()

    # Legal transitions:
    #  _RUNNING -> _DEBUGGABLE \
    #           -> _CRASHED ----\
    #           -> _FINISHED ----\
    #           ------------------+-> _CLOSED

    def __init__(self, fn, iterators, kwargs, special_args):
        self._fn = fn
        self._kwargs = kwargs
        self._iterators = iterators
        self._iterators_have_more = True
        self._special_args = special_args

        if _config["processes"] == "auto":
            processes = multiprocessing.cpu_count()
        else:
            processes = _config["processes"]

        self._worker_ids = itertools.count()
        self._workers = {}
        # This is a made-up number.
        self._work_queue = multiprocessing.Queue(2 * processes)
        # This number has some theory. For full-throughput operation, the
        # result queue needs to be large enough to handle all worker processes
        # completing a job at the same time without blocking. But the only way
        # to get more results than that onto the queue at the same time would
        # be for at least one worker to complete two jobs in the time it less
        # than the time it takes the parent to dequeue one job, which means
        # that the the workers are going faster than the parent so they should
        # probably slow down. So blocking is appropriate, and having exactly
        # 'processes' slots in the queue is probably about right.
        self._result_queue = multiprocessing.Queue(processes)
        # We never debug more than one process, so no reason for a larger
        # queue.
        self._debug_start_queue = multiprocessing.Queue(1)
        self._debug_done_queue = multiprocessing.Queue(1)

        self._state = self._RUNNING
        # If in _DEBUGGABLE state, the worker that failed (kept around so we can
        # debug it if requested)
        self._debuggable_worker = None
        # Count of how many jobs we have put on work_queue
        self._jobs_started = 0
        # Count of how many jobs we have taken off result_queue
        self._jobs_finished = 0

        for i in xrange(processes):
            self._spawn_worker()

        self._turn_crank()

    def _spawn_worker(self):
        assert self._state is self._RUNNING
        worker_id = self._worker_ids.next()
        args = (worker_id,
                self._fn, self._kwargs,
                self._work_queue, self._result_queue,
                self._debug_start_queue, self._debug_done_queue)
        worker = multiprocessing.Process(target=_mpimap_worker, args=args)
        self._workers[worker_id] = worker
        worker.start()

    def __iter__(self):
        return self

    def next(self):
        if (self._state is self._RUNNING
            and not self._iterators_have_more
            and self._jobs_finished == self._jobs_started):
            self._transition(self._FINISHED)
        if self._state is self._DEBUGGABLE:
            raise ParimapError("called next() on a mapper that has already "
                              "failed")
        elif self._state is self._CRASHED:
            raise ParimapError("child process died unexpectedly")
        elif self._state is self._CLOSED:
            raise ParimapError(".next() called on a closed iterator")
        elif self._state is self._FINISHED:
            raise StopIteration
        else:
            assert self._state is self._RUNNING
            self._turn_crank()
            result = self._result_queue.get()
            self._jobs_finished += 1
            if result[0] == "success":
                return result[1]
            elif result[0] == "exc":
                _, exc, tb_list, worker_id = result
                self._transition(self._DEBUGGABLE, worker_id)
                wrapped_exc = _try_wrap_exception(exc, tb_list, self)
                raise wrapped_exc
            else:
                assert False

    def _turn_crank(self):
        # Fill the output queue. We might as well be greedy here, and it's
        # okay that workers are pulling work off the queue at the same time
        # that we add it -- all this work is work that has to happen sooner or
        # later, in serial, in this process. If the workers go faster than the
        # input iterator, then eventually this loop will still terminate once
        # the work and result queues have both filled up.
        if self._iterators_have_more:
            # .full() is documented to be somewhat unreliable, but the code
            # looks clear to me -- it just checks the value of the semaphore
            # that counts queued items, and put() acquires this semaphore
            # first thing. So after a put() it will necessarily have increased
            # unless semaphore. Maybe they're just warning about cases where
            # on thread of execution calls put() simultaneously with another
            # calling full().
            while not self._work_queue.full():
                try:
                    values = [it.next() for it in self._iterators]
                except StopIteration:
                    self._iterators_have_more = False
                    break
                self._jobs_started += 1
                self._work_queue.put(values)
        # Check for dead processes
        # XX FIXME: should we key this off SIGCHLD or a timer or anything? it
        # might be a bit expensive to do a full poll after every single job.
        for worker in self._workers.itervalues():
            if not worker.is_alive():
                self._transition(self._CRASHED)
                return

    def post_mortem(self):
        if self._state is not self._DEBUGGABLE:
            raise ParimapError("no jobs have failed, so nothing to debug")
        self._debug_start_queue.put(None)
        result = self._debug_done_queue.get()
        if result[0] == "debug-exc":
            raise result[1]
        elif result[0] == "debug-success":
            return result[1]
        else:
            assert False

    def _transition(self, new_state, debuggable_worker_id=None):
        if new_state is self._state:
            return
        # Anything can transition to _CLOSED. Nothing can transition to
        # _RUNNING. Everything else can only transition *from* _RUNNING. All
        # states except for _RUNNING are failure states in which we kill off
        # all workers.
        assert new_state is not self._RUNNING
        if new_state is not self._CLOSED:
            assert self._state is self._RUNNING
        if new_state is self._DEBUGGABLE:
            assert debuggable_worker_id is not None
            assert self._debuggable_worker is None
            self._debuggable_worker = self._workers[debuggable_worker_id]
            del self._workers[debuggable_worker_id]
        else:
            assert debuggable_worker_id is None
            if self._debuggable_worker is not None:
                self._debuggable_worker.terminate()
                self._debuggable_worker.join()
                self._debuggable_worker = None
        for worker in self._workers.itervalues():
            worker.terminate()
        for worker in self._workers.itervalues():
            worker.join()
        self._workers.clear()
        self._state = new_state

    def close(self):
        self._transition(self._CLOSED)

    def __del__(self):
        if hasattr(self, "_state"):
            self.close()
