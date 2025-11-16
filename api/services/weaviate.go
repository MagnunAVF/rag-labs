package services

import (
	"context"
	"errors"
	"fmt"

	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/graphql"
	"github.com/weaviate/weaviate/entities/models"
)

type WeaviateService struct {
	client         *weaviate.Client
	collectionName string
	searchLimit    int
}

func NewWeaviateService(client *weaviate.Client, collectionName string, searchLimit int) *WeaviateService {
	return &WeaviateService{
		client:         client,
		collectionName: collectionName,
		searchLimit:    searchLimit,
	}
}

func (s *WeaviateService) SearchSimilar(ctx context.Context, embedding []float64) ([]string, error) {
	vector32 := make([]float32, len(embedding))
	for i, v := range embedding {
		vector32[i] = float32(v)
	}

	fields := []graphql.Field{{Name: "text"}}
	nearVector := s.client.GraphQL().
		NearVectorArgBuilder().
		WithVector(vector32)

	resp, err := s.client.GraphQL().
		Get().
		WithClassName(s.collectionName).
		WithFields(fields...).
		WithNearVector(nearVector).
		WithLimit(s.searchLimit).
		Do(ctx)
	if err != nil {
		return nil, fmt.Errorf("Weaviate query failed: %w", err)
	}

	return extractTextChunks(resp.Data, s.collectionName)
}

func extractTextChunks(data map[string]models.JSONObject, collectionName string) ([]string, error) {
	get, ok := data["Get"].(map[string]interface{})
	if !ok {
		return nil, errors.New("invalid GraphQL response: missing Get field")
	}

	class, ok := get[collectionName].([]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid GraphQL response: missing collection %s", collectionName)
	}

	var chunks []string
	for _, item := range class {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		text, ok := itemMap["text"].(string)
		if ok && text != "" {
			chunks = append(chunks, text)
		}
	}

	return chunks, nil
}
