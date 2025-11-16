package config

import (
	"os"
	"time"
)

const (
	DefaultVLLMBaseURL    = "http://localhost:8000/v1"
	DefaultTEIBaseURL     = "http://localhost:8082"
	DefaultVLLMModelName  = "microsoft/Phi-3-mini-128k-instruct"
	DefaultWeaviateHost   = "localhost:8080"
	DefaultWeaviateScheme = "http"
	DefaultCollectionName = "LlamaIndex"
	DefaultServerPort     = ":8081"
	DefaultMaxTokens      = 4096
	DefaultSearchLimit    = 3
	DefaultHTTPTimeout    = 30 * time.Second
	ShutdownTimeout       = 10 * time.Second
)

type Config struct {
	VLLMBaseURL    string
	TEIBaseURL     string
	VLLMModelName  string
	WeaviateHost   string
	WeaviateScheme string
	CollectionName string
	ServerPort     string
	MaxTokens      int
	SearchLimit    int
}

func Load() *Config {
	return &Config{
		VLLMBaseURL:    getEnv("VLLM_BASE_URL", DefaultVLLMBaseURL),
		TEIBaseURL:     getEnv("TEI_BASE_URL", DefaultTEIBaseURL),
		VLLMModelName:  getEnv("VLLM_MODEL_NAME", DefaultVLLMModelName),
		WeaviateHost:   getEnv("WEAVIATE_HOST", DefaultWeaviateHost),
		WeaviateScheme: getEnv("WEAVIATE_SCHEME", DefaultWeaviateScheme),
		CollectionName: getEnv("COLLECTION_NAME", DefaultCollectionName),
		ServerPort:     getEnv("SERVER_PORT", DefaultServerPort),
		MaxTokens:      DefaultMaxTokens,
		SearchLimit:    DefaultSearchLimit,
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
