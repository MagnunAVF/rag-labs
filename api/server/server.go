package server

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"rag-api/config"
	"rag-api/handlers"
	"rag-api/models"
	"rag-api/services"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/sashabaranov/go-openai"
	"github.com/weaviate/weaviate-go-client/v4/weaviate"
)

type Server struct {
	app        *fiber.App
	cfg        *config.Config
	ragService *services.RAGService
}

func New(cfg *config.Config) (*Server, error) {
	openaiConfig := openai.DefaultConfig("DUMMY_API_KEY")
	openaiConfig.BaseURL = cfg.VLLMBaseURL
	openaiClient := openai.NewClientWithConfig(openaiConfig)

	httpClient := &http.Client{
		Timeout: config.DefaultHTTPTimeout,
	}

	weaviateCfg := weaviate.Config{
		Host:   cfg.WeaviateHost,
		Scheme: cfg.WeaviateScheme,
	}
	weaviateClient, err := weaviate.NewClient(weaviateCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create Weaviate client: %w", err)
	}

	log.Printf("Initialized clients - vLLM: %s, TEI: %s, Weaviate: %s",
		cfg.VLLMBaseURL, cfg.TEIBaseURL, cfg.WeaviateHost)

	teiService := services.NewTEIService(cfg.TEIBaseURL, httpClient)
	weaviateService := services.NewWeaviateService(weaviateClient, cfg.CollectionName, cfg.SearchLimit)
	vllmService := services.NewVLLMService(openaiClient, cfg.VLLMModelName, cfg.MaxTokens)
	ragService := services.NewRAGService(teiService, weaviateService, vllmService)

	app := setupApp(ragService)

	return &Server{
		app:        app,
		cfg:        cfg,
		ragService: ragService,
	}, nil
}

func setupApp(ragService *services.RAGService) *fiber.App {
	app := fiber.New(fiber.Config{
		ErrorHandler: customErrorHandler,
		AppName:      "RAG API Server",
	})

	app.Use(recover.New())
	app.Use(logger.New(logger.Config{
		Format:     "${time} ${status} - ${method} ${path} (${latency})\n",
		TimeFormat: "2006-01-02 15:04:05",
	}))
	app.Use(cors.New())

	queryHandler := handlers.NewQueryHandler(ragService)
	app.Post("/query", queryHandler.Handle)
	app.Get("/health", handlers.HandleHealth)

	return app
}

func customErrorHandler(c *fiber.Ctx, err error) error {
	code := fiber.StatusInternalServerError
	if e, ok := err.(*fiber.Error); ok {
		code = e.Code
	}
	log.Printf("Error: %v", err)
	return c.Status(code).JSON(models.ErrorResponse{Error: err.Error()})
}

func (s *Server) Start() error {
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("Starting RAG API server on %s", s.cfg.ServerPort)
		if err := s.app.Listen(s.cfg.ServerPort); err != nil {
			log.Printf("Server error: %v", err)
		}
	}()

	<-quit
	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), config.ShutdownTimeout)
	defer cancel()

	if err := s.app.ShutdownWithContext(ctx); err != nil {
		return fmt.Errorf("server forced to shutdown: %w", err)
	}

	log.Println("Server exited gracefully")
	return nil
}
