package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/imcanugur/ai-model-runtime/internal/domain"
	internalHTTP "github.com/imcanugur/ai-model-runtime/internal/http"
	"github.com/imcanugur/ai-model-runtime/internal/jobstore"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	storeRedis "github.com/imcanugur/ai-model-runtime/internal/jobstore/redis"
	"github.com/imcanugur/ai-model-runtime/internal/pipeline"
	"github.com/imcanugur/ai-model-runtime/internal/queue"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	queueRedis "github.com/imcanugur/ai-model-runtime/internal/queue/redis"
	"github.com/imcanugur/ai-model-runtime/internal/registry"
	"github.com/imcanugur/ai-model-runtime/internal/runtime"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/mock"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/onnx"
	"github.com/imcanugur/ai-model-runtime/internal/runtime/safetensors"
	"github.com/imcanugur/ai-model-runtime/internal/service"
	"github.com/imcanugur/ai-model-runtime/internal/worker"
	"github.com/redis/go-redis/v9"
)

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func main() {
	portStr := flag.String("port", getEnv("PORT", "8080"), "HTTP server port")
	host := flag.String("host", getEnv("HOST", "0.0.0.0"), "HTTP server host")
	modelsDir := flag.String("models-dir", getEnv("MODELS_DIR", "./models"), "Path to models directory")
	redisAddr := flag.String("redis-addr", getEnv("REDIS_ADDR", ""), "Redis address for queue and jobstore (leave empty for in-memory)")
	workerCountStr := flag.String("workers", getEnv("WORKERS", "4"), "Number of asynchronous background workers")
	defaultModelID := flag.String("default-model", getEnv("DEFAULT_MODEL", "text-generation-tr"), "Default model ID for compatibility endpoints")
	flag.Parse()

	port, err := strconv.Atoi(*portStr)
	if err != nil {
		port = 8080
	}
	workerCount, err := strconv.Atoi(*workerCountStr)
	if err != nil {
		workerCount = 4
	}

	log.Printf("=================================================================")
	log.Printf("  AI Model Runtime (100%% Pure Go Engine)")
	log.Printf("=================================================================")
	log.Printf("Listening Address   : %s:%d", *host, port)
	log.Printf("Swagger UI          : http://127.0.0.1:%d/swagger/", port)
	log.Printf("Models Directory    : %s", *modelsDir)
	log.Printf("Engine Mode         : Pure Go (SafeTensors, ONNX, Mock)")
	log.Printf("Redis Storage       : %s", func() string {
		if *redisAddr != "" {
			return *redisAddr
		}
		return "Disabled (Using In-Memory Store & Queue)"
	}())
	log.Printf("Worker Pool Size    : %d", workerCount)
	log.Printf("Default Model ID    : %s", *defaultModelID)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 1. Initialize Pure-Go Runtime Providers & Factory
	rtFactory := runtime.NewDefaultFactory()
	_ = rtFactory.RegisterProvider("mock", mock.NewProvider())
	_ = rtFactory.RegisterProvider("safetensors", safetensors.NewProvider())
	_ = rtFactory.RegisterProvider("native", safetensors.NewProvider())
	_ = rtFactory.RegisterProvider("onnx", onnx.NewProvider(nil))

	// 2. Initialize Stage Registry & Model Registry
	stageReg := pipeline.NewDefaultStageRegistry()
	modelReg := registry.NewDefaultModelRegistry(stageReg, registry.AdaptRuntimeFactory(rtFactory))

	// 3. Discover Models
	cleanModelsDir, _ := filepath.Abs(*modelsDir)
	log.Printf("Scanning models from %s ...", cleanModelsDir)
	if err := modelReg.Discover(ctx, cleanModelsDir); err != nil {
		log.Printf("Warning during model discovery: %v", err)
	}

	// 4. Pre-load Discovered Models
	descriptors := modelReg.List()
	log.Printf("Discovered %d model package(s):", len(descriptors))
	for _, desc := range descriptors {
		log.Printf("  -> Found [%s] (Task: %s, State: %s)", desc.Key, desc.Metadata.Task, desc.State)
		if desc.State == domain.StateDiscovered {
			if loadErr := modelReg.Load(ctx, desc.Key); loadErr != nil {
				log.Printf("     [WARN] Failed to load model %s: %v", desc.Key, loadErr)
			} else {
				log.Printf("     [OK] Model %s loaded and READY", desc.Key)
			}
		}
	}

	// 5. Initialize JobStore and Queue
	var store jobstore.JobStore
	var q queue.JobQueue

	if *redisAddr != "" {
		log.Printf("Connecting to Redis at %s ...", *redisAddr)
		rdb := redis.NewClient(&redis.Options{Addr: *redisAddr})
		if err := rdb.Ping(ctx).Err(); err != nil {
			log.Fatalf("Fatal: Failed to connect to Redis at %s: %v", *redisAddr, err)
		}
		var err error
		store, err = storeRedis.NewRedisJobStore(storeRedis.RedisJobStoreConfig{
			Client:    rdb,
			KeyPrefix: "ai-model-runtime:jobs:",
			TTL:       24 * time.Hour,
		})
		if err != nil {
			log.Fatalf("Fatal: Failed to initialize Redis job store: %v", err)
		}
		q, err = queueRedis.NewRedisQueue(queueRedis.RedisQueueConfig{
			Client:    rdb,
			KeyPrefix: "ai-model-runtime:queue:",
		})
		if err != nil {
			log.Fatalf("Fatal: Failed to initialize Redis queue: %v", err)
		}
	} else {
		store = storeMemory.NewMemoryJobStore()
		q = queueMemory.NewMemoryQueue()
	}

	// 6. Initialize Worker Pool
	pool, err := worker.NewWorkerPool(worker.WorkerPoolConfig{
		Count:          workerCount,
		Queue:          q,
		Store:          store,
		Registry:       modelReg,
		DequeueTimeout: 500 * time.Millisecond,
	})
	if err != nil {
		log.Fatalf("Fatal: Failed to initialize worker pool: %v", err)
	}
	if err := pool.Start(ctx); err != nil {
		log.Fatalf("Fatal: Failed to start worker pool: %v", err)
	}

	// 7. Initialize Inference Service
	inferSvc, err := service.NewInferenceService(service.ServiceConfig{
		Registry: modelReg,
		JobStore: store,
		Queue:    q,
	})
	if err != nil {
		log.Fatalf("Fatal: Failed to initialize inference service: %v", err)
	}

	// 8. Initialize HTTP Server with Swagger & Endpoints
	serverCfg := internalHTTP.DefaultServerConfig()
	serverCfg.Address = fmt.Sprintf("%s:%d", *host, port)
	serverCfg.DefaultModelID = domain.ModelID(*defaultModelID)

	server, err := internalHTTP.NewServer(serverCfg, inferSvc, internalHTTP.ServerDependencies{
		WorkerPool: pool,
		Queue:      q,
	})
	if err != nil {
		log.Fatalf("Fatal: Failed to initialize HTTP server: %v", err)
	}

	// 9. Run HTTP Listener in Background
	httpServer := &http.Server{
		Addr:    fmt.Sprintf("%s:%d", *host, port),
		Handler: server.Handler(),
	}

	go func() {
		log.Printf("AI Model Runtime HTTP Server is READY and listening at http://%s:%d", *host, port)
		log.Printf("Visit Swagger UI documentation at http://127.0.0.1:%d/swagger/", port)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Fatal: HTTP server listen error: %v", err)
		}
	}()

	// 10. Handle Graceful Shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	sig := <-sigChan
	log.Printf("Received termination signal (%v). Commencing graceful shutdown...", sig)

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("Error shutting down HTTP server: %v", err)
	}

	_ = pool.Stop(shutdownCtx)
	_ = q.Close()

	// Unload loaded models cleanly
	for _, desc := range modelReg.List() {
		if desc.State == domain.StateReady {
			_ = modelReg.Unload(shutdownCtx, desc.Key)
		}
	}

	log.Printf("AI Model Runtime shutdown completed cleanly.")
}
