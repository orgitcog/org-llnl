// Single-producer/multiple-consumer queue
// Copyright (C) 2021, Lawrence Livermore National Security, LLC.
//  All rights reserved. LLNL-CODE-837067
//
// The Department of Homeland Security sponsored the production of this
//  material under DOE Contract Number DE-AC52-07N427344 for the management
//  and operation of Lawrence Livermore National Laboratory. Contract no.
//  DE-AC52-07NA27344 is between the U.S. Department of Energy (DOE) and
//  Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
//  See license for disclaimers, notice of U.S. Government Rights and license
//  terms and conditions.

namespace tmon
{

template <typename T>
SpmcQueue<T>::SpmcQueue()
    : num_consumers_{}, mutex_{}, cond_{}
{
}

template <typename T>
SpmcQueue<T>::SpmcQueue(int num_consumers)
    : num_consumers_{num_consumers}, mutex_{}, cond_{}
{
    head_for_consumer_.resize(num_consumers_, 0);
}

template <typename T>
SpmcQueue<T>::~SpmcQueue()
{
    cancel();
}

template <typename T>
int SpmcQueue<T>::add_consumer()
{
    unique_lock<mutex> guard{mutex_};
    head_for_consumer_.push_back(0);
    for (auto& x : cont_)
        ++x.potential_consumers;
    return ++num_consumers_;
}

// Cancel any blocking waits on this queue (by signaling conditional)
template <typename T>
void SpmcQueue<T>::cancel()
{
    cond_.notify_all();
}

template <typename T>
class SpmcQueue<T>::Listener
{
  public:
    using MsgType = T;

    Listener(SpmcQueue& queue, int consumer_idx)
        : queue_{queue}, consumer_idx_{consumer_idx}
    {
    }
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    Listener(Listener&&) = default;
    Listener& operator=(Listener&&) = default;

    void cancel()
    {
        queue_.cancel();
    }

    optional<T> pop_front()
    {
        return queue_.pop_front(consumer_idx_);
    }

    [[nodiscard]] bool empty() const
    {
        return queue_.empty(consumer_idx_);
    }

    template <typename Predicate>
    bool wait(Predicate fn) const
    {
        return queue_.wait(consumer_idx_, fn);
    }

  private:
    SpmcQueue& queue_;
    int consumer_idx_;
};

template <typename T>
auto SpmcQueue<T>::get_new_listener() -> Listener
{
    int new_idx = add_consumer();
    return {*this, new_idx};
}

// Pushes a copy of an element to the queue, with a tag indicating that
//  none of the consumers have yet popped it
template <typename T>
void SpmcQueue<T>::push_back(T x)
{
    {
    unique_lock<mutex> guard{mutex_};
    cont_.push_back({std::move(x), num_consumers_});
    }
    cond_.notify_all();
}

// Returns a copy of the front element in the queue if there may be other
//  consumers yet to read it; otherwise, pops it and returns by move
template <typename T>
optional<T> SpmcQueue<T>::pop_front(int consumer)
{
    unique_lock<mutex> guard{mutex_};
    assert((consumer > 0) && (consumer <= head_for_consumer_.size()));

    int head_idx = head_for_consumer_[consumer - 1];
    if (head_idx == cont_.size())
        return {}; // empty
    // Since this function will, by this point, either return a value by copy or
    //  move, increment the head index for this consumer
    ++head_for_consumer_[consumer - 1];
    assert((head_idx >= 0) && (head_idx < cont_.size()));
    if (--cont_[head_idx].potential_consumers > 0)
        return cont_[head_idx].data;
    // No other potential consumers for this element; must be the front of
    //  the queue or something has gone awry
    assert((head_idx == 0) && "spmc should be popping front of queue");
    T popped{std::move(cont_.front().data)};
    cont_.pop_front();
    for (int& x : head_for_consumer_)
        --x;
    return popped;
}

// Whether or not the queue is empty for the specified consumer (which may,
//  but does not necessarily, imply that the queue is empty altogether)
template <typename T>
[[nodiscard]] bool SpmcQueue<T>::empty(int consumer) const
{
    unique_lock<mutex> guard{mutex_};
    return unsynchronized_empty(consumer);
}

// Private implementation for empty() intended to be called under lock on mutex_
template <typename T>
[[nodiscard]] bool SpmcQueue<T>::unsynchronized_empty(int consumer) const
{
    assert((consumer > 0) && (consumer <= head_for_consumer_.size()));
    if (cont_.empty())
        return true;
    int head_idx = head_for_consumer_[consumer - 1];
    assert((head_idx >= 0) && (head_idx <= cont_.size()));
    return (head_idx == cont_.size());
}

template <typename T>
void SpmcQueue<T>::reserve(std::size_t sz)
{
    cont_.reserve(sz);
}

// Waits until data is available for the specified consumer (i.e., empty
//  would return false) or the provided predicate is satisfied; returns true
//  if data is available
template <typename T>
template <typename Predicate>
bool SpmcQueue<T>::wait(int consumer, Predicate fn) const
{
    unique_lock<mutex> guard{mutex_};
    cond_.wait(guard, [this, consumer, &fn]{
            return !this->unsynchronized_empty(consumer) || fn(); });
    return !unsynchronized_empty(consumer);
}

} // end namespace tmon

// Copyright (C) 2021, Lawrence Livermore National Security, LLC.
//  All rights reserved. LLNL-CODE-837067
//
// The Department of Homeland Security sponsored the production of this
//  material under DOE Contract Number DE-AC52-07N427344 for the management
//  and operation of Lawrence Livermore National Laboratory. Contract no.
//  DE-AC52-07NA27344 is between the U.S. Department of Energy (DOE) and
//  Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
//  See license for disclaimers, notice of U.S. Government Rights and license
//  terms and conditions.


