package worker

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/jobstore"
	"github.com/imcanugur/ai-model-runtime/internal/queue"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
)

// PoolState defines the lifecycle state of the WorkerPool.
type PoolState int32

const (
	PoolInit PoolState = iota
	PoolRunning
	PoolStopping
	PoolStopped
)

// WorkerPoolConfig configures a concurrent pool of workers.
type WorkerPoolConfig struct {
	Count           int
	Queue           queue.JobQueue
	Store           jobstore.JobStore
	Registry        registry.ModelRegistry
	LeaseDuration   time.Duration
	HeartbeatPeriod time.Duration
	DequeueTimeout  time.Duration
	BackoffBase     time.Duration
	BackoffMax      time.Duration
	Sleeper         Sleeper
	Observer        JobObserver
}

// WorkerPool coordinates multiple Worker goroutines.
type WorkerPool struct {
	cfg     WorkerPoolConfig
	workers []*Worker

	state  int32 // PoolState atomic
	wg     sync.WaitGroup
	stopCh chan struct{}
	cancel context.CancelFunc
	mu     sync.Mutex
}

// NewWorkerPool instantiates a WorkerPool with the specified configuration.
func NewWorkerPool(cfg WorkerPoolConfig) (*WorkerPool, error) {
	if cfg.Count <= 0 {
		cfg.Count = runtime.NumCPU()
		if cfg.Count < 2 {
			cfg.Count = 2
		}
	}
	if cfg.Queue == nil {
		return nil, errors.New("queue cannot be nil")
	}
	if cfg.Store == nil {
		return nil, errors.New("store cannot be nil")
	}
	if cfg.Registry == nil {
		return nil, errors.New("registry cannot be nil")
	}

	workers := make([]*Worker, 0, cfg.Count)
	for i := 0; i < cfg.Count; i++ {
		workerID := fmt.Sprintf("worker-%d-%d", time.Now().UnixNano()%10000, i+1)
		w, err := NewWorker(WorkerConfig{
			ID:              workerID,
			Queue:           cfg.Queue,
			Store:           cfg.Store,
			Registry:        cfg.Registry,
			LeaseDuration:   cfg.LeaseDuration,
			HeartbeatPeriod: cfg.HeartbeatPeriod,
			DequeueTimeout:  cfg.DequeueTimeout,
			BackoffBase:     cfg.BackoffBase,
			BackoffMax:      cfg.BackoffMax,
			Sleeper:         cfg.Sleeper,
			Observer:        cfg.Observer,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create worker %d: %w", i+1, err)
		}
		workers = append(workers, w)
	}

	return &WorkerPool{
		cfg:     cfg,
		workers: workers,
		state:   int32(PoolInit),
		stopCh:  make(chan struct{}),
	}, nil
}

// Start spawns worker goroutines and begins job execution.
func (p *WorkerPool) Start(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !atomic.CompareAndSwapInt32(&p.state, int32(PoolInit), int32(PoolRunning)) {
		return errors.New("worker pool already started or closed")
	}

	runCtx, cancel := context.WithCancel(ctx)
	p.cancel = cancel

	for _, w := range p.workers {
		p.wg.Add(1)
		go p.runWorker(runCtx, w)
	}

	return nil
}

func (p *WorkerPool) runWorker(ctx context.Context, w *Worker) {
	defer p.wg.Done()

	for {
		// Check pool stop signal
		if atomic.LoadInt32(&p.state) != int32(PoolRunning) {
			return
		}

		select {
		case <-p.stopCh:
			return
		case <-ctx.Done():
			return
		default:
		}

		// Process one job. If queue is empty or timed out, it simply loops.
		_, err := w.ProcessOne(ctx)
		if err != nil {
			// Worker error isolation: individual job errors or transient dequeue errors do NOT crash the pool
			select {
			case <-ctx.Done():
				return
			case <-p.stopCh:
				return
			default:
			}
		}
	}
}

// Stop initiates graceful shutdown: stops accepting new jobs and waits for in-flight tasks to finish.
func (p *WorkerPool) Stop(ctx context.Context) error {
	p.mu.Lock()
	if !atomic.CompareAndSwapInt32(&p.state, int32(PoolRunning), int32(PoolStopping)) {
		p.mu.Unlock()
		return nil // Already stopping or stopped
	}

	close(p.stopCh)
	if p.cancel != nil {
		p.cancel()
	}
	p.mu.Unlock()

	// Wait for active workers to drain
	done := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		atomic.StoreInt32(&p.state, int32(PoolStopped))
		return nil
	case <-ctx.Done():
		atomic.StoreInt32(&p.state, int32(PoolStopped))
		return fmt.Errorf("worker pool graceful stop timed out: %w", ctx.Err())
	}
}

// State returns current PoolState.
func (p *WorkerPool) State() PoolState {
	return PoolState(atomic.LoadInt32(&p.state))
}

// Workers returns the slice of initialized workers in the pool.
func (p *WorkerPool) Workers() []*Worker {
	return p.workers
}
